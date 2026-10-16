"""Defer a Telegram update to Cloud Tasks so the webhook can return immediately."""

from __future__ import annotations

import json
import logging

from gae_platform import config

logger = logging.getLogger(__name__)


def enqueue_update(body: dict) -> bool:
    if not config.use_cloud_tasks():
        return False
    project = config.project_id()
    if not project:
        logger.warning("PROJECT_ID is not set; processing the update inline")
        return False
    try:
        from google.cloud import tasks_v2
    except ImportError:
        logger.warning("google-cloud-tasks is not installed; processing inline")
        return False
    try:
        client = tasks_v2.CloudTasksClient()
        parent = client.queue_path(project, config.app_region(), config.cloud_tasks_queue())
        # https on purpose: an http appspot URL answers 301 and the task body is dropped.
        task = {
            "http_request": {
                "http_method": tasks_v2.HttpMethod.POST,
                "url": f"https://{project}.appspot.com{config.process_task_path()}",
                "headers": {"Content-Type": "application/json"},
                "body": json.dumps(body).encode(),
            }
        }
        client.create_task(parent=parent, task=task)
    except Exception:
        logger.exception("Cloud Task enqueue failed; processing inline")
        return False
    logger.info("Scheduled Cloud Task for update %s", body.get("update_id"))
    return True


def enqueue_rating_continue(chain: int) -> bool:
    """Queue the next rating batch. The URL is not logged: it contains the webhook secret."""
    if not config.use_cloud_tasks() or not config.obfuscation_token():
        return False
    project = config.project_id()
    if not project:
        logger.warning("PROJECT_ID is not set; rating continuation waits for the next cron")
        return False
    try:
        from google.cloud import tasks_v2
    except ImportError:
        logger.warning("google-cloud-tasks is not installed; rating continuation waits for the next cron")
        return False
    try:
        client = tasks_v2.CloudTasksClient()
        parent = client.queue_path(project, config.app_region(), config.cloud_tasks_queue())
        task = {
            "http_request": {
                "http_method": tasks_v2.HttpMethod.POST,
                "url": f"https://{project}.appspot.com{config.rating_continue_path()}",
                "headers": {"Content-Type": "application/json"},
                "body": json.dumps({"chain": chain}).encode(),
            }
        }
        client.create_task(parent=parent, task=task)
    except Exception:
        logger.exception("Rating continuation enqueue failed")
        return False
    logger.info("Scheduled rating continuation %s", chain)
    return True
