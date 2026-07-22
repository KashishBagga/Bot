#!/usr/bin/env python3
import os
import sys
import json
import logging
from datetime import datetime, date
from urllib.parse import urlparse, parse_qs
from http.server import HTTPServer, BaseHTTPRequestHandler

# Path Injection
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from src.models.postgres_database import PostgresDatabase

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("DashboardServer")

db = PostgresDatabase()

class DashboardHandler(BaseHTTPRequestHandler):
    def end_headers(self):
        # Allow CORS
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'X-Requested-With, Content-Type')
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()

    def do_GET(self):
        parsed_url = urlparse(self.path)
        path = parsed_url.path
        query = parse_qs(parsed_url.query)

        # 1. API: List available report dates
        if path == "/api/reports":
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            
            reports_dir = os.path.join(project_root, "reports")
            dates = []
            if os.path.exists(reports_dir):
                for f in os.listdir(reports_dir):
                    if f.endswith(".json") and not f.startswith("daily_"):
                        dates.append(f.replace(".json", ""))
            
            # Sort descending
            dates.sort(reverse=True)
            self.wfile.write(json.dumps(dates).encode('utf-8'))
            return

        # 2. API: Get detailed report data for a date
        elif path == "/api/report":
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()

            report_date = query.get("date", [date.today().strftime('%Y-%m-%d')])[0]
            
            response_data = {
                "date": report_date,
                "eod_report": None,
                "trades": [],
                "candidates": [],
                "execution_events": []
            }

            # 2.1 Load static EOD JSON report
            report_file = os.path.join(project_root, "reports", f"{report_date}.json")
            if os.path.exists(report_file):
                try:
                    with open(report_file, 'r') as f:
                        response_data["eod_report"] = json.load(f)
                except Exception as e:
                    logger.error(f"Error loading report file: {e}")

            # 2.2 Query PostgreSQL database
            try:
                with db._get_connection() as conn:
                    # Fetch real trades
                    with conn.cursor() as cursor:
                        cursor.execute("""
                            SELECT trade_id, candidate_id, entry_time, exit_time, symbol, strategy, 
                                   entry_price, exit_price, pnl, exit_reason, mfe_r, mae_r, final_pnl_r, 
                                   bars_held, stop_loss, take_profit, experiment_name, diagnostics, features
                            FROM trade_performance
                            WHERE entry_time::date = %s
                            ORDER BY entry_time ASC
                        """, (report_date,))
                        columns = [desc[0] for desc in cursor.description]
                        response_data["trades"] = [
                            dict(zip(columns, [self.serialize_field(val) for val in row]))
                            for row in cursor.fetchall()
                        ]

                    # Fetch candidates (rejections / counterfactual results)
                    with conn.cursor() as cursor:
                        cursor.execute("""
                            SELECT candidate_id, timestamp, symbol, signal_type, setup_type, 
                                   rejection_reasons, primary_rejection_reason, entry_price, 
                                   stop_loss, take_profit, exit_time, exit_price, mfe_r, mae_r, final_pnl_r, 
                                   experiment_name, diagnostics
                            FROM counterfactual_results
                            WHERE timestamp::date = %s
                            ORDER BY timestamp ASC
                        """, (report_date,))
                        columns = [desc[0] for desc in cursor.description]
                        response_data["candidates"] = [
                            dict(zip(columns, [self.serialize_field(val) for val in row]))
                            for row in cursor.fetchall()
                        ]

                    # Fetch execution events for latency tracking
                    with conn.cursor() as cursor:
                        cursor.execute("""
                            SELECT event_id, trade_id, candidate_id, timestamp, event_type, payload
                            FROM execution_events
                            WHERE timestamp::date = %s
                            ORDER BY timestamp ASC
                        """, (report_date,))
                        columns = [desc[0] for desc in cursor.description]
                        response_data["execution_events"] = [
                            dict(zip(columns, [self.serialize_field(val) for val in row]))
                            for row in cursor.fetchall()
                        ]

            except Exception as e:
                logger.error(f"Error querying Postgres for dashboard data: {e}", exc_info=True)

            self.wfile.write(json.dumps(response_data).encode('utf-8'))
            return

        # 3. Serve EOD Dashboard UI HTML
        elif path == "/" or path == "/index.html":
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.end_headers()
            
            html_file = os.path.join(project_root, "src", "trading", "dashboard_web", "index.html")
            if os.path.exists(html_file):
                with open(html_file, 'rb') as f:
                    self.wfile.write(f.read())
            else:
                self.wfile.write(b"<h1>EOD Dashboard UI not found in src/trading/dashboard_web/index.html</h1>")
            return

        # 4. Handle other paths
        else:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"Not Found")

    def serialize_field(self, val):
        """Prepares values for JSON serialization (datetime to ISO strings, dicts/lists to python structures)."""
        if isinstance(val, (datetime, date)):
            return val.isoformat()
        if isinstance(val, (dict, list)):
            return val
        return val

def run(port=8080):
    server_address = ('', port)
    httpd = HTTPServer(server_address, DashboardHandler)
    logger.info(f"⚡ EOD Dashboard Web Server running at: http://localhost:{port}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        logger.info("🛑 Shutting down server...")
        httpd.server_close()

if __name__ == "__main__":
    port = 8080
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except ValueError:
            pass
    run(port)
