import json
import logging
import uuid
from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import field
from datetime import datetime
from datetime import timezone
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_TITLE = "New chat"
MAX_TITLE_LENGTH = 60


class ConversationNotFoundError(KeyError):
    pass


@dataclass
class ConversationTurnRecord:
    role: str
    content: str
    created_at: str


@dataclass
class Conversation:
    conversation_id: str
    user_id: str
    title: str
    created_at: str
    updated_at: str
    turns: list[ConversationTurnRecord] = field(default_factory=list)


class ConversationStore:
    """
    S3-backed per-user chat history for the multi-turn /ask flow - same
    "S3 instead of a new database" pattern as IngestionJobStore/
    S3DocumentStore/S3BackupTarget. Keys are namespaced by user_id
    (conversations/{user_id}/{conversation_id}.json), which is what
    actually enforces ownership: every read/write requires the caller's
    own authenticated user_id (the OIDC token's `sub` claim) as part of
    the S3 key, never a client-supplied one - a caller structurally
    cannot address another user's conversation regardless of what
    conversation_id they pass.

    The server is the only writer of turn content (see the /ask route -
    it always appends the real query and the real generated answer
    itself); there is no API that lets a client write arbitrary
    "assistant" text into a conversation's history.
    """

    def __init__(
        self,
        client: Any,
        bucket_name: str,
        prefix: str = "conversations/"
    ) -> None:
        self.client = client
        self.bucket_name = bucket_name
        self.prefix = prefix

    def create_conversation(
        self,
        user_id: str,
        title: str = DEFAULT_TITLE
    ) -> Conversation:
        now = self._now()
        conversation = Conversation(
            conversation_id=str(uuid.uuid4()),
            user_id=user_id,
            title=title,
            created_at=now,
            updated_at=now,
            turns=[]
        )
        self._write(conversation)
        logger.info(
            "conversation_created",
            extra={"conversation_id": conversation.conversation_id, "user_id": user_id}
        )
        return conversation

    def get_conversation(
        self,
        conversation_id: str,
        user_id: str
    ) -> Conversation | None:
        try:
            response = self.client.get_object(
                Bucket=self.bucket_name, Key=self._key(user_id, conversation_id)
            )
        except self.client.exceptions.NoSuchKey:
            return None

        data = json.loads(response["Body"].read())
        return self._from_dict(data)

    def append_turns(
        self,
        conversation_id: str,
        user_id: str,
        new_turns: list[tuple[str, str]]
    ) -> Conversation:
        """
        new_turns: list of (role, content) pairs, appended in order.
        Raises ConversationNotFoundError if the conversation doesn't
        exist or isn't owned by user_id - callers should have already
        confirmed ownership via get_conversation() before generating an
        answer, but this re-checks rather than trusting that.
        """
        conversation = self.get_conversation(conversation_id, user_id)

        if conversation is None:
            raise ConversationNotFoundError(conversation_id)

        now = self._now()

        for role, content in new_turns:
            conversation.turns.append(ConversationTurnRecord(role=role, content=content, created_at=now))

        if conversation.title == DEFAULT_TITLE:
            first_user_message = next((content for role, content in new_turns if role == "user"), None)

            if first_user_message:
                conversation.title = first_user_message[:MAX_TITLE_LENGTH]

        conversation.updated_at = now
        self._write(conversation)
        return conversation

    def list_conversations(
        self,
        user_id: str
    ) -> list[Conversation]:
        """
        Most-recently-updated first. Fetches each conversation object
        individually rather than maintaining a separate index - simpler,
        and correct at the per-user conversation counts this app
        actually has (a handful to dozens, not thousands).
        """
        user_prefix = f"{self.prefix}{user_id}/"
        paginator = self.client.get_paginator("list_objects_v2")
        conversations = []

        for page in paginator.paginate(Bucket=self.bucket_name, Prefix=user_prefix):
            for obj in page.get("Contents", []):
                conversation_id = obj["Key"].rsplit("/", 1)[-1].removesuffix(".json")
                conversation = self.get_conversation(conversation_id, user_id)

                if conversation is not None:
                    conversations.append(conversation)

        conversations.sort(key=lambda c: c.updated_at, reverse=True)
        return conversations

    def delete_conversation(
        self,
        conversation_id: str,
        user_id: str
    ) -> None:
        self.client.delete_object(Bucket=self.bucket_name, Key=self._key(user_id, conversation_id))
        logger.info(
            "conversation_deleted",
            extra={"conversation_id": conversation_id, "user_id": user_id}
        )

    def _write(
        self,
        conversation: Conversation
    ) -> None:
        self.client.put_object(
            Bucket=self.bucket_name,
            Key=self._key(conversation.user_id, conversation.conversation_id),
            Body=json.dumps(self._to_dict(conversation)).encode("utf-8"),
            ContentType="application/json"
        )

    def _key(
        self,
        user_id: str,
        conversation_id: str
    ) -> str:
        return f"{self.prefix}{user_id}/{conversation_id}.json"

    def _to_dict(
        self,
        conversation: Conversation
    ) -> dict[str, Any]:
        return asdict(conversation)

    def _from_dict(
        self,
        data: dict[str, Any]
    ) -> Conversation:
        return Conversation(
            conversation_id=data["conversation_id"],
            user_id=data["user_id"],
            title=data["title"],
            created_at=data["created_at"],
            updated_at=data["updated_at"],
            turns=[ConversationTurnRecord(**turn) for turn in data.get("turns", [])]
        )

    def _now(self) -> str:
        return datetime.now(timezone.utc).isoformat()
