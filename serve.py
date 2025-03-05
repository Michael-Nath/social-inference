import http.server
import ssl

PORT = 3000  # Change if needed
certfile = "/etc/letsencrypt/live/socialinference.duckdns.org/fullchain.pem"
keyfile  = "/etc/letsencrypt/live/socialinference.duckdns.org/privkey.pem"

server_address = ("0.0.0.0", PORT)
handler = http.server.SimpleHTTPRequestHandler

httpd = http.server.HTTPServer(server_address, handler)

context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
context.load_cert_chain(certfile=certfile, keyfile=keyfile)

httpd.socket = context.wrap_socket(httpd.socket, server_side=True)

print(f"Serving HTTPS on port {PORT}...")
httpd.serve_forever()
