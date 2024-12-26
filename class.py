import pandas as pd

# 读取原始CSV文件
input_file = './csv_output/Tunnel/l3-total-add.csv'

# 读取CSV文件（文件中已经有列名）
df = pd.read_csv(input_file)

# 根据Label列进行过滤，创建两个不同的DataFrame
doh_tunnel_traffic_hkd_df = df[df['Label'].isin(['tcp-over-dns', 'dnstt', 'tuns'])]
cira_cic_dohbrw_2020_df = df[df['Label'].isin(['dns2tcp', 'iodine', 'dnscat2'])]

# 将过滤后的DataFrame保存为新的CSV文件
doh_tunnel_traffic_hkd_df.to_csv('./csv_output/Tunnel/doh-tunnel-traffic-hkd.csv', index=False)
cira_cic_dohbrw_2020_df.to_csv('./csv_output/Tunnel/cira-cic-dohbrw-2020.csv', index=False)

print("Files have been split and saved.")
