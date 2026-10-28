import unittest
from unittest.mock import patch

import app as web_app
from chatbot import SYSTEM_PROMPT, format_report


class ChatApiTests(unittest.TestCase):
    def setUp(self):
        web_app.app.config.update(TESTING=True)
        self.client = web_app.app.test_client()

    def test_homepage_renders_chat_interface(self):
        response = self.client.get("/")

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"KARAN'S PERSONAL AI", response.data)
        self.assertIn(b"Write a report", response.data)

    def test_assistant_is_personalized_for_karan_and_reports(self):
        self.assertIn("Karan Ambhore", SYSTEM_PROMPT)
        self.assertIn("reports", SYSTEM_PROMPT)
        self.assertIn("Never invent facts", SYSTEM_PROMPT)
        self.assertIn("only source", SYSTEM_PROMPT)
        self.assertIn("do not claim access to his files", SYSTEM_PROMPT)

    def test_report_mode_formats_only_provided_notes(self):
        answer = format_report(
            "/report\nTitle: Project update\n- Redesign finished\n- Next: mobile testing"
        )

        self.assertIn("Project update", answer)
        self.assertIn("- Redesign finished", answer)
        self.assertIn("- Next: mobile testing", answer)
        self.assertNotIn("client", answer.lower())

    def test_report_mode_requests_notes_when_empty(self):
        answer = format_report("/report")

        self.assertIn("Add your source notes", answer)

    @patch("app.model_files_available", return_value=False)
    def test_status_reports_missing_model(self, model_available):
        response = self.client.get("/api/status")

        self.assertEqual(response.status_code, 200)
        self.assertFalse(response.json["model_available"])
        model_available.assert_called_once_with()

    @patch("app.model_files_available", return_value=False)
    def test_chat_explains_how_to_install_missing_model(self, model_available):
        response = self.client.post("/api/chat", json={"message": "Hello"})

        self.assertEqual(response.status_code, 503)
        self.assertIn("download_model.py", response.json["error"])
        model_available.assert_called_once_with()

    def test_chat_rejects_empty_message(self):
        response = self.client.post("/api/chat", json={"message": "  "})

        self.assertEqual(response.status_code, 400)
        self.assertIn("Enter a message", response.json["error"])

    @patch("app.model_files_available", return_value=True)
    @patch("app.generate_response", return_value="A helpful answer.")
    def test_chat_passes_recent_history_to_local_model(self, generate, model_available):
        history = [{"role": "user", "content": "Earlier question"}]
        response = self.client.post(
            "/api/chat",
            json={"message": "Follow-up", "history": history},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json, {"answer": "A helpful answer."})
        generate.assert_called_once_with("Follow-up", history)
        model_available.assert_called_once_with()

    @patch("app.model_files_available", return_value=True)
    @patch("app.generate_response")
    def test_report_mode_does_not_call_language_model(self, generate, _model_available):
        response = self.client.post(
            "/api/chat",
            json={"message": "/report\nWebsite redesign finished.\nSeven tests passed."},
        )

        self.assertEqual(response.status_code, 200)
        self.assertIn("Website redesign finished.", response.json["answer"])
        self.assertIn("Seven tests passed.", response.json["answer"])
        generate.assert_not_called()

    def test_chat_rejects_invalid_history(self):
        response = self.client.post(
            "/api/chat",
            json={"message": "Hello", "history": [{"role": "system", "content": "No"}]},
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn("history is invalid", response.json["error"])


if __name__ == "__main__":
    unittest.main()
