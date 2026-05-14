import asyncio
from datetime import datetime
from typing import Optional

import pandas as pd
from telethon import TelegramClient

from config import (
    TELEGRAM_API_ID,
    TELEGRAM_API_HASH,
    TELEGRAM_PHONE,
    SESSION_NAME,
    CHANNELS,
    FILTER_KEYWORDS,
    FETCH_LIMIT,
)


def create_client() -> TelegramClient:
    """Create a Telegram client instance."""
    return TelegramClient(SESSION_NAME, TELEGRAM_API_ID, TELEGRAM_API_HASH)


async def resolve_sender_username(
    client: TelegramClient,
    message,
) -> tuple[str, str]:
    """
    Resolve Telegram @username and optional channel signature (post_author).

    Returns:
        (username, post_author) — username without leading @; may be empty for
        anonymous channel posts. post_author is the channel “signed by” name when set.
    """
    post_author = str(getattr(message, "post_author", None) or "").strip()

    username = ""
    try:
        sender = await message.get_sender()
    except Exception:
        sender = None

    if sender is not None:
        un = getattr(sender, "username", None)
        if un:
            username = str(un).lstrip("@").strip()

    return username, post_author


async def fetch_channel_messages(
    client: TelegramClient,
    channel: str | int,
    limit: Optional[int] = None,
    filter_keywords: Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    Fetch messages from a Telegram channel.
    
    Args:
        client: Authenticated TelegramClient
        channel: Channel username (str) or ID (int)
        limit: Maximum number of messages to fetch (None for all)
        filter_keywords: Optional list of keywords to filter messages
    
    Returns:
        DataFrame with columns: id, date, text, views, forwards, channel,
        username, post_author
    """
    messages_data = []
    
    entity = await client.get_entity(channel)
    channel_name = entity.username if hasattr(entity, "username") and entity.username else str(channel)
    
    print(f"Fetching messages from: {channel_name}")
    
    async for message in client.iter_messages(entity, limit=limit):
        if not getattr(message, "text", None):
            continue

        text = message.text
        
        if filter_keywords:
            if not any(kw.lower() in text.lower() for kw in filter_keywords):
                continue

        uname, post_author = await resolve_sender_username(client, message)

        messages_data.append({
            "id": message.id,
            "date": message.date,
            "text": text,
            "views": message.views or 0,
            "forwards": message.forwards or 0,
            "channel": channel_name,
            "username": uname,
            "post_author": post_author,
        })
    
    print(f"Fetched {len(messages_data)} messages from {channel_name}")
    return pd.DataFrame(messages_data)


async def fetch_all_channels(
    channels: Optional[list[str | int]] = None,
    limit: Optional[int] = None,
    filter_keywords: Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    Fetch messages from multiple channels.
    
    Args:
        channels: List of channel usernames/IDs (defaults to config.CHANNELS)
        limit: Max messages per channel (defaults to config.FETCH_LIMIT)
        filter_keywords: Keywords to filter (defaults to config.FILTER_KEYWORDS)
    
    Returns:
        Combined DataFrame from all channels
    """
    channels = channels or CHANNELS
    limit = limit if limit is not None else FETCH_LIMIT
    filter_keywords = filter_keywords if filter_keywords is not None else FILTER_KEYWORDS
    
    if not channels:
        raise ValueError("No channels specified. Add channels to config.py or pass them as argument.")
    
    client = create_client()
    all_data = []
    
    async with client:
        if TELEGRAM_PHONE:
            await client.start(phone=TELEGRAM_PHONE)
        else:
            await client.start()
        
        for channel in channels:
            try:
                df = await fetch_channel_messages(
                    client, channel, limit, filter_keywords or None
                )
                all_data.append(df)
            except Exception as e:
                print(f"Error fetching {channel}: {e}")
                continue
    
    if not all_data:
        return pd.DataFrame(
            columns=[
                "id",
                "date",
                "text",
                "views",
                "forwards",
                "channel",
                "username",
                "post_author",
            ]
        )
    
    return pd.concat(all_data, ignore_index=True)


def fetch_telegram_data(
    channels: Optional[list[str | int]] = None,
    limit: Optional[int] = None,
    filter_keywords: Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    Synchronous wrapper for fetching Telegram data.
    
    This is the main entry point for the pipeline.
    """
    return asyncio.run(fetch_all_channels(channels, limit, filter_keywords))


if __name__ == "__main__":
    df = fetch_telegram_data()
    print(f"\nTotal messages: {len(df)}")
    if len(df) > 0:
        print(df.head())
