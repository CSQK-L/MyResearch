import os

from django.core.management.base import BaseCommand, CommandError

from khoj.database.models import ChatModel
from khoj.utils.constants import default_deepseek_chat_models
from khoj.utils.initialization import _setup_chat_model_provider


class Command(BaseCommand):
    help = "Register DeepSeek text chat models without changing defaults or existing provider settings."

    def handle(self, *args, **options):
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key or not api_key.strip():
            raise CommandError("DEEPSEEK_API_KEY is required.")
        try:
            _setup_chat_model_provider(
                ChatModel.ModelType.DEEPSEEK,
                default_deepseek_chat_models,
                default_api_key=api_key,
                api_base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
                vision_enabled=False,
                interactive=False,
                provider_name="DeepSeek",
            )
        except ValueError as exc:
            raise CommandError(str(exc)) from None
        self.stdout.write(self.style.SUCCESS("DeepSeek chat models registered; existing selections preserved."))
