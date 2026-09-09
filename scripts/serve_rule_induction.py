# Mục đích: phục vụ giao diện Rule Induction và lưu kết quả người dùng xác nhận.
# Chức năng: nhận kết quả theo từng PDF qua HTTP và ghi vào một file JSON.
# Luồng xử lý: mở thư mục output → phục vụ HTML → nhận POST /confirm → lưu JSON.
# Cấu hình/parameter: nhận đường dẫn thư mục chứa rule_induction.html.
# Quan hệ: phục vụ output do test_rule_induction_page3_plus.py tạo ra.
# Input/output: đọc nội dung HTTP; ghi confirmed_results.json trong thư mục output.
# Contract: POST /confirm nhận một phiếu có pdf_name và ghi theo khóa pdf_name.

import argparse
import json
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


HOST = "127.0.0.1"
PORT = 8000


class RuleInductionRequestHandler(SimpleHTTPRequestHandler):
    """Phục vụ giao diện và lưu kết quả Rule Induction đã xác nhận."""

    results_path = None

    def do_POST(self):
        """Lưu kết quả của một phiếu khi nhận yêu cầu POST /confirm."""
        if self.path != "/confirm":
            self.send_error(404)
            return

        content_length = int(self.headers["Content-Length"])
        confirmed_form = json.loads(
            self.rfile.read(content_length).decode("utf-8")
        )
        if self.results_path.is_file():
            confirmed_results = json.loads(
                self.results_path.read_text(encoding="utf-8")
            )
        else:
            confirmed_results = {
                "schema_version": 1,
                "forms": {},
            }

        confirmed_results["forms"][confirmed_form["pdf_name"]] = confirmed_form
        self.results_path.write_text(
            json.dumps(
                confirmed_results,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        response_body = json.dumps({"saved": True}).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(response_body)))
        self.end_headers()
        self.wfile.write(response_body)


def parse_arguments():
    """Đọc thư mục output cần phục vụ."""
    parser = argparse.ArgumentParser()
    parser.add_argument("output_directory", type=Path)
    return parser.parse_args()


def main():
    """Chạy HTTP server cục bộ cho giao diện Rule Induction."""
    arguments = parse_arguments()
    output_directory = arguments.output_directory.resolve()
    RuleInductionRequestHandler.results_path = (
        output_directory / "confirmed_results.json"
    )
    request_handler = partial(
        RuleInductionRequestHandler,
        directory=output_directory,
    )
    server = ThreadingHTTPServer(
        (HOST, PORT),
        request_handler,
    )
    print(f"Giao diện: http://{HOST}:{PORT}/rule_induction.html")
    print(f"Kết quả xác nhận: {RuleInductionRequestHandler.results_path}")
    server.serve_forever()


if __name__ == "__main__":
    main()
