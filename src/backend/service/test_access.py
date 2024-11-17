import socket


def test_connection(ip, port):
    try:
        with socket.create_connection((ip, port), timeout=10) as sock:
            return True
    except Exception as e:
        return False, str(e)


# Define the Qdrant VM IP and ports
qdrant_ip = "34.142.165.16"  # Use internal IP if accessing from within the same VPC
ports = [6333, 6334]

# Test the connection to Qdrant
results = []
for port in ports:
    result = test_connection(qdrant_ip, port)
    if isinstance(result, tuple):
        results.append((port, False, result[1]))
    else:
        results.append((port, True))

for port, success, *error in results:
    if success:
        print(f"Port {port}: Connection successful")
    else:
        print(f"Port {port}: Connection failed - {error[0]}")
