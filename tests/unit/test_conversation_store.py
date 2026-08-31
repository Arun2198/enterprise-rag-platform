import itertools

import pytest

from app.conversation_store import ConversationNotFoundError
from app.conversation_store import ConversationStore


class FakeBody:

    def __init__(self, data: bytes):
        self._data = data

    def read(self):
        return self._data


class NoSuchKey(Exception):
    pass


class FakeExceptions:
    NoSuchKey = NoSuchKey


class FakePaginator:

    def __init__(self, client):
        self.client = client

    def paginate(self, Bucket, Prefix):
        keys = [key for key in self.client.objects if key.startswith(Prefix)]
        yield {"Contents": [{"Key": key} for key in keys]}


class FakeS3Client:

    def __init__(self):
        self.objects: dict[str, bytes] = {}
        self.exceptions = FakeExceptions()

    def put_object(self, Bucket, Key, Body, ContentType=None):
        self.objects[Key] = Body

    def get_object(self, Bucket, Key):
        if Key not in self.objects:
            raise self.exceptions.NoSuchKey()

        return {"Body": FakeBody(self.objects[Key])}

    def delete_object(self, Bucket, Key):
        self.objects.pop(Key, None)

    def get_paginator(self, name):
        assert name == "list_objects_v2"
        return FakePaginator(self)


def test_create_conversation_defaults_to_new_chat_title():
    store = ConversationStore(client=FakeS3Client(), bucket_name="my-bucket")

    conversation = store.create_conversation("user-1")

    assert conversation.title == "New chat"
    assert conversation.turns == []


def test_get_conversation_returns_none_when_missing():
    store = ConversationStore(client=FakeS3Client(), bucket_name="my-bucket")

    assert store.get_conversation("missing", "user-1") is None


def test_get_conversation_scoped_to_owning_user():
    store = ConversationStore(client=FakeS3Client(), bucket_name="my-bucket")
    conversation = store.create_conversation("user-1")

    # a different user id can never address this conversation, regardless
    # of knowing the real conversation_id - ownership is the S3 key itself
    assert store.get_conversation(conversation.conversation_id, "user-2") is None
    assert store.get_conversation(conversation.conversation_id, "user-1") is not None


def test_append_turns_persists_and_updates_title_from_first_user_message():
    store = ConversationStore(client=FakeS3Client(), bucket_name="my-bucket")
    conversation = store.create_conversation("user-1")

    updated = store.append_turns(
        conversation.conversation_id,
        "user-1",
        [("user", "What are the vacation days?"), ("assistant", "Full-time employees get 15 days.")]
    )

    assert updated.title == "What are the vacation days?"
    assert [turn.role for turn in updated.turns] == ["user", "assistant"]
    assert updated.updated_at >= conversation.created_at


def test_append_turns_on_unknown_conversation_raises():
    store = ConversationStore(client=FakeS3Client(), bucket_name="my-bucket")

    with pytest.raises(ConversationNotFoundError):
        store.append_turns("missing", "user-1", [("user", "hi")])


def test_list_conversations_returns_only_the_caller_own_and_sorted_by_recency():
    store = ConversationStore(client=FakeS3Client(), bucket_name="my-bucket")
    # force strictly increasing timestamps - real wall-clock calls can land
    # in the same microsecond on a fast machine, which would make the sort
    # a no-op and defeat the point of this test.
    clock = itertools.count()
    store._now = lambda: f"2026-01-01T00:00:{next(clock):02d}Z"

    first = store.create_conversation("user-1", title="First chat")
    second = store.create_conversation("user-1", title="Second chat")
    store.create_conversation("user-2", title="Someone else's chat")

    # bump second's updated_at ahead of first's so recency ordering is real
    store.append_turns(second.conversation_id, "user-1", [("user", "hi")])

    conversations = store.list_conversations("user-1")

    assert [c.conversation_id for c in conversations] == [
        second.conversation_id, first.conversation_id
    ]


def test_delete_conversation_removes_it():
    store = ConversationStore(client=FakeS3Client(), bucket_name="my-bucket")
    conversation = store.create_conversation("user-1")

    store.delete_conversation(conversation.conversation_id, "user-1")

    assert store.get_conversation(conversation.conversation_id, "user-1") is None
