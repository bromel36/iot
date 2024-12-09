import pandas as pd
import subprocess

# Bước 1: Trích xuất thông tin packet từ packets.pcap
def extract_packet_info(pcap_file, output_csv):
    # Tạo lệnh tshark
    tshark_cmd = [
        "tshark", "-r", pcap_file, "-T", "fields",
        "-e", "frame.number", "-e", "ip.src", "-e", "tcp.srcport",
        "-e", "ip.dst", "-e", "tcp.dstport", "-e", "ip.proto",
        "-E", "header=y", "-E", "separator=,", "-E", "quote=d", "-E", "occurrence=f"
    ]
    # Chạy lệnh tshark
    with open(output_csv, "w") as outfile:
        subprocess.run(tshark_cmd, stdout=outfile, check=True)

# Bước 2: Đọc dữ liệu từ các file CSV
def label_flows(packet_info_file, packet_labels_file, flow_file, output_file):
    # Đọc thông tin packet
    packet_info = pd.read_csv(packet_info_file)
    packet_labels = pd.read_csv(packet_labels_file)
    flows = pd.read_csv(flow_file)

    # Ghép nhãn từ file packets.csv vào thông tin packet
    packet_info = packet_info.rename(columns={"frame.number": "PacketNumber"})
    packet_info = packet_info.merge(packet_labels, on="PacketNumber", how="left")

    # Hàm kiểm tra packet và flow có khớp hay không
    def is_matching_flow(flow, packet):
        return (
            (flow["Src IP"] == packet["ip.src"] and flow["Src Port"] == packet["tcp.srcport"] and
             flow["Dst IP"] == packet["ip.dst"] and flow["Dst Port"] == packet["tcp.dstport"] and
             flow["Protocol"] == packet["ip.proto"]) or
            (flow["Src IP"] == packet["ip.dst"] and flow["Src Port"] == packet["tcp.dstport"] and
             flow["Dst IP"] == packet["ip.src"] and flow["Dst Port"] == packet["tcp.srcport"] and
             flow["Protocol"] == packet["ip.proto"])
        )

    # Hàm gán nhãn cho một flow dựa trên các packets
    def label_flow(flow, packets):
        # Lọc các packet khớp với flow
        matching_packets = packets.apply(lambda p: is_matching_flow(flow, p), axis=1)
        # Nếu bất kỳ packet nào có nhãn "ARP-Spoofing", thì flow cũng có nhãn này
        if "ACK" in packets[matching_packets]["Label"].values:
            return "1"
        return "0"

    # Gán nhãn cho tất cả các flow
    flows["Label"] = flows.apply(lambda f: label_flow(f, packet_info), axis=1)

    # Lưu kết quả ra file CSV
    flows.to_csv(output_file, index=False)
    print(f"Phân nhãn hoàn tất. Kết quả đã được lưu vào '{output_file}'.")

# Đường dẫn file
pcap_file = "../pcaps/ACK_v3.pcap"
packet_info_csv = "packet_info.csv"
packet_labels_csv = "ack_labelled.csv"
flows_csv = "./csv_flow/ACK_v3.pcap_Flow.csv"
output_labeled_flows_csv = "./csv_flow/ACK_v3.pcap_Flow.csv"

# Thực hiện
extract_packet_info(pcap_file, packet_info_csv)
print(f"Thông tin packet được lưu tại: {packet_info_csv}")


label_flows(packet_info_csv, packet_labels_csv, flows_csv, output_labeled_flows_csv)
