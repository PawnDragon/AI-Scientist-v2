import os
import unittest
from types import SimpleNamespace
from unittest import mock

from ai_scientist import model_providers


class OpenClawGatewayProviderTest(unittest.TestCase):
    def test_openai_api_provider_preserves_sdk_construction(self):
        with mock.patch("openai.OpenAI") as openai_cls:
            client = model_providers.create_openai_client(
                provider=model_providers.OPENAI_API_PROVIDER,
                max_retries=3,
            )

        self.assertEqual(client, openai_cls.return_value)
        openai_cls.assert_called_once_with(max_retries=3)

    def test_openclaw_gateway_provider_uses_gateway_url_and_token(self):
        with mock.patch.dict(
            os.environ,
            {
                model_providers.OPENCLAW_BASE_URL_ENV: "http://127.0.0.1:18789",
                model_providers.OPENCLAW_GATEWAY_TOKEN_ENV: "gateway-token",
            },
            clear=True,
        ):
            with mock.patch("openai.OpenAI") as openai_cls:
                client = model_providers.create_openai_client(
                    provider=model_providers.OPENCLAW_GATEWAY_PROVIDER,
                    max_retries=1,
                )

        self.assertIsInstance(client, model_providers.OpenClawGatewayClient)
        openai_cls.assert_called_once_with(
            max_retries=1,
            api_key="gateway-token",
            base_url="http://127.0.0.1:18789/v1",
        )

    def test_openclaw_gateway_maps_model_to_header(self):
        completions = mock.Mock()
        sdk_client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=completions))
        )
        client = model_providers.OpenClawGatewayClient(sdk_client)

        client.chat.completions.create(
            model="openai/gpt-5.5",
            messages=[{"role": "user", "content": "OK?"}],
        )

        completions.assert_called_once_with(
            model="openclaw/default",
            messages=[{"role": "user", "content": "OK?"}],
            extra_headers={"x-openclaw-model": "openai/gpt-5.5"},
        )

    def test_openclaw_gateway_can_disable_model_header_mapping(self):
        completions = mock.Mock()
        sdk_client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=completions))
        )
        client = model_providers.OpenClawGatewayClient(sdk_client)

        with mock.patch.dict(
            os.environ,
            {model_providers.OPENCLAW_USE_MODEL_HEADER_ENV: "0"},
            clear=True,
        ):
            client.chat.completions.create(
                model="openai/gpt-5.5",
                messages=[{"role": "user", "content": "OK?"}],
            )

        completions.assert_called_once_with(
            model="openai/gpt-5.5",
            messages=[{"role": "user", "content": "OK?"}],
        )


if __name__ == "__main__":
    unittest.main()
