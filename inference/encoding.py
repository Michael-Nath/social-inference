def read_be_int(offset: int, data: bytes) -> tuple[int, int]:
    return int.from_bytes(data[offset:offset+4], byteorder='big'), offset + 4

def write_be_int(data: bytearray, offset: int, value: int) -> int:
    data[offset:offset+4] = value.to_bytes(4, byteorder='big')
    return offset + 4

def read_encoded_string(offset: int, data: bytes) -> tuple[str, int]:
    length, offset = read_be_int(offset, data)
    return data[offset:offset+length].decode('utf-8'), offset + length

def write_encoded_string(data: bytearray, offset: int, string: str) -> int:
    offset = write_be_int(data, offset, len(string))
    data[offset:offset+len(string)] = string.encode('utf-8')
    return offset + len(string)

def size_encoded_string(string: str) -> int:
    return len(string.encode('utf-8')) + 4 # 4 bytes for the length

