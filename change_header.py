import pandas as pd


def harmonize_column_names(file1, file2):
    """
    Đồng nhất tên cột của file2 theo tên cột của file1 và lưu kết quả.

    Args:
    - file1 (str): Đường dẫn tới tệp CSV đầu tiên (chuẩn).
    - file2 (str): Đường dẫn tới tệp CSV thứ hai cần đổi tên cột.
    - output_file (str): Đường dẫn để lưu tệp CSV đầu ra.

    Returns:
    - None
    """
    # Đọc hai tệp CSV
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)

    # Lấy tên cột từ file1
    columns_file1 = list(df1.columns)

    # Đổi tên cột của file2
    df2.columns = columns_file1

    # Lưu file đã đổi tên cột
    df2.to_csv(file2, index=False)
    print(f"Đã lưu file với tên cột đồng nhất tại: ")


# Ví dụ sử dụng:
file1_path = './csv_flow/dos-synflooding-1-dec.pcap_Flow.csv'  # Đường dẫn tới file1
file2_path = './csv_flow/MITM ARP Spoofing_test.csv'  # Đường dẫn tới file2
output_path = './csv_flow/MITM ARP Spoofing_test.csv'  # Đường dẫn lưu file sau khi đổi tên cột

harmonize_column_names(file1_path, file2_path)
