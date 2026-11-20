from dataclasses import dataclass

from django.utils.translation import gettext, ngettext

from management.models import FileObject, ServiceAccount, Team, Token


@dataclass(frozen=True)
class Crumb:
    label: str
    url: str = ""
    mono: bool = False


@dataclass(frozen=True)
class Confirmation:
    title: str
    detail: str
    action: str


def deletion(obj: Token | ServiceAccount | Team | FileObject) -> Confirmation:
    match obj:
        case Token():
            name = obj.name
            detail = gettext(
                "Apps using it stop working, and its requests, files, batches, and vector stores "
                "are deleted."
            )
        case ServiceAccount():
            name = obj.name
            detail = gettext(
                "Its key stops working, and the key's requests, files, batches, and vector stores "
                "are deleted."
            )
        case Team():
            name, count = obj.name, obj.service_accounts.count()
            if count:
                detail = ngettext(
                    "Its %(count)s service account is deleted with its key and the key's "
                    "requests, files, batches, and vector stores. Members keep their personal "
                    "keys.",
                    "Its %(count)s service accounts are deleted with their keys and the keys' "
                    "requests, files, batches, and vector stores. Members keep their personal "
                    "keys.",
                    count,
                ) % {"count": count}
            else:
                detail = gettext("Members keep their personal keys.")
        case FileObject():
            name = obj.filename
            detail = gettext(
                "Batches that use it as input are deleted with it, and vector stores drop it."
            )
        case _:
            raise TypeError(f"no deletion text for {type(obj).__name__}")
    return Confirmation(gettext("Delete %(name)s?") % {"name": name}, detail, gettext("Delete"))
