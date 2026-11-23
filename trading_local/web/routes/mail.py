"""E-mail alerts: test mail and the log of all mail attempts."""

from collections.abc import Callable
from typing import Annotated

from fastapi import APIRouter, Query

from trading_local.db import repo_ops
from trading_local.dto import MailLogOut, MailTestOut
from trading_local.presenters import address_masker, mask_email
from trading_local.services import Services
from trading_local.web.deps import Ctx

router = APIRouter(prefix="/api")


@router.post("/mail/test")
async def send_test_mail(ctx: Ctx) -> MailTestOut:
    return await ctx.services.mailer.send_test()


@router.get("/mail/log")
async def list_mail_log(ctx: Ctx, limit: Annotated[int, Query(ge=1, le=500)] = 100) -> list[MailLogOut]:
    return await ctx.run(_masked_mail_log, ctx.services, limit)


def _masked_mail_log(services: Services, limit: int) -> list[MailLogOut]:
    """Mail attempts, newest first, with masked recipients (addresses never leave the process in full).

    The details are masked as well: SMTP replies often repeat the sender or recipient address. The mailer
    masks them before it stores a detail; this second pass covers rows written by older versions.
    """
    settings = services.settings
    configured = (settings.mail_to, settings.sender, settings.smtp_user)
    maskers: dict[str, Callable[[str], str]] = {}  # by the row's recipient (usually one value for all rows)
    out: list[MailLogOut] = []
    for entry in repo_ops.list_mail_log(services.db, limit=limit):
        mask_details = maskers.get(entry.recipient)
        if mask_details is None:
            mask_details = maskers[entry.recipient] = address_masker((*configured, entry.recipient))
        out.append(entry.model_copy(update={"recipient": mask_email(entry.recipient), "detail": mask_details(entry.detail)}))
    return out
