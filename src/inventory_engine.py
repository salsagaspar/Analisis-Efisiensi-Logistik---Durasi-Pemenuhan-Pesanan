"""
src/inventory_engine.py
Sistem Optimasi Reorder Point (ROP), Safety Stock Dinamis (King's Formula),
dan Generator Rekomendasi Purchase Order (PO) Otomatis.

Dibuat untuk Supply Chain Ritel Vintage 90an (Shopee).
"""

import os
import argparse
import numpy as np
import pandas as pd
from scipy.stats import norm

class InventoryOptimizationEngine:
    def __init__(self, excel_path: str, holding_cost_rate: float = 0.20, order_cost: float = 50000.0):
        self.excel_path = excel_path
        self.holding_cost_rate = holding_cost_rate
        self.order_cost = order_cost
        
        self.df_prod = None
        self.df_supp = None
        self.df_po = None
        self.df_ord = None
        self.df_inv = None
        self.result_df = None
        
    def load_data(self):
        """Memuat 5 tabel data inti dari Excel."""
        if not os.path.exists(self.excel_path):
            raise FileNotFoundError(f"File dataset tidak ditemukan: {self.excel_path}")
            
        xl = pd.ExcelFile(self.excel_path)
        self.df_prod = xl.parse('Produk_Master')
        self.df_supp = xl.parse('Pemasok')
        self.df_po = xl.parse('Pembelian')
        self.df_ord = xl.parse('Pesanan')
        self.df_inv = xl.parse('Persediaan')
        return self

    def run_pipeline(self) -> pd.DataFrame:
        """Menjalankan pipeline kalkulasi demand, lead time, ABC-XYZ, SS dinamis, dan ROP."""
        if self.df_prod is None:
            self.load_data()
            
        # 1. Hitung profil permintaan harian per SKU
        date_range = pd.date_range(start=self.df_ord['Tanggal_Order'].min(), end=self.df_ord['Tanggal_Order'].max(), freq='D')
        all_skus = self.df_prod['SKU'].unique()
        grid_df = pd.DataFrame(index=pd.MultiIndex.from_product([date_range, all_skus], names=['Tanggal_Order', 'SKU'])).reset_index()

        daily_actual = self.df_ord.groupby(['Tanggal_Order', 'SKU'])['Qty'].sum().reset_index()
        daily_full = pd.merge(grid_df, daily_actual, on=['Tanggal_Order', 'SKU'], how='left').fillna({'Qty': 0})
        
        demand_stats = daily_full.groupby('SKU').agg(
            Demand_Mean_Harian=('Qty', 'mean'),
            Demand_Std_Harian=('Qty', 'std'),
            Total_Terjual=('Qty', 'sum')
        ).reset_index()
        demand_stats['Demand_CV'] = demand_stats['Demand_Std_Harian'] / demand_stats['Demand_Mean_Harian']
        
        # 2. Hitung statistik performa supplier
        po_supp = pd.merge(self.df_po, self.df_supp[['Supplier_ID', 'Nama_Pemasok', 'Lead_Time_Target_Hari', 'MOQ_Unit']], on='Supplier_ID')
        overall_lt_mean = self.df_po['Lead_Time_Aktual_Hari'].mean()
        overall_lt_std = self.df_po['Lead_Time_Aktual_Hari'].std()
        
        supp_stats = po_supp.groupby('Supplier_ID').agg(
            LT_Mean_Aktual=('Lead_Time_Aktual_Hari', 'mean'),
            LT_Std_Aktual=('Lead_Time_Aktual_Hari', 'std'),
            LT_Target=('Lead_Time_Target_Hari', 'first'),
            MOQ_Unit=('MOQ_Unit', 'first')
        ).reset_index()
        supp_stats['LT_Std_Aktual'] = supp_stats['LT_Std_Aktual'].fillna(overall_lt_std)
        
        # 3. Gabungkan ke master tabel
        res = self.df_prod.copy()
        res = res.merge(demand_stats, on='SKU', how='left')
        res = res.merge(supp_stats, on='Supplier_ID', how='left')
        
        inv_sub = self.df_inv[['SKU', 'Stok_Akhir', 'Safety_Stock_Unit', 'Reorder_Point_Unit', 'Status_Stok']].rename(
            columns={
                'Safety_Stock_Unit': 'Safety_Stock_Eksisting',
                'Reorder_Point_Unit': 'ROP_Eksisting',
                'Status_Stok': 'Status_Stok_Eksisting'
            }
        )
        res = res.merge(inv_sub, on='SKU', how='left')
        
        # 4. Segmentasi ABC-XYZ & Penentuan Service Level
        res['Omzet_Rp'] = res['Total_Terjual'] * res['Harga_Jual_Rp']
        res = res.sort_values(by='Omzet_Rp', ascending=False).reset_index(drop=True)
        res['Omzet_Kumulatif_%'] = (res['Omzet_Rp'] / res['Omzet_Rp'].sum()).cumsum() * 100
        
        res['Kelas_ABC'] = res['Omzet_Kumulatif_%'].apply(lambda x: 'A' if x <= 80 else ('B' if x <= 95 else 'C'))
        res['Kelas_XYZ'] = res['Demand_CV'].apply(lambda cv: 'X' if cv < 0.5 else ('Y' if cv <= 1.0 else 'Z'))
        res['Segmen_ABC_XYZ'] = res['Kelas_ABC'] + res['Kelas_XYZ']
        
        service_level_map = {
            'AX': 0.98, 'AY': 0.98, 'AZ': 0.95,
            'BX': 0.95, 'BY': 0.93, 'BZ': 0.90,
            'CX': 0.90, 'CY': 0.88, 'CZ': 0.85
        }
        res['Service_Level_Target'] = res['Segmen_ABC_XYZ'].map(service_level_map).fillna(0.95)
        res['Z_Score'] = res['Service_Level_Target'].apply(lambda sl: norm.ppf(sl))
        
        # 5. Formula King's Safety Stock
        res['LT_Mean_Clean'] = res['LT_Mean_Aktual'].fillna(overall_lt_mean)
        res['LT_Std_Clean'] = res['LT_Std_Aktual'].fillna(overall_lt_std)
        
        term_d_var = res['LT_Mean_Clean'] * (res['Demand_Std_Harian'] ** 2)
        term_lt_var = (res['Demand_Mean_Harian'] ** 2) * (res['LT_Std_Clean'] ** 2)
        variance_combined = term_d_var + term_lt_var
        
        res['Safety_Stock_Dinamis'] = np.ceil(res['Z_Score'] * np.sqrt(variance_combined)).astype(int)
        
        # 6. ROP Dinamis
        res['ROP_Dinamis'] = np.ceil((res['Demand_Mean_Harian'] * res['Lead_Time_Target_Hari']) + res['Safety_Stock_Dinamis']).astype(int)
        
        # Guardrail untuk One-off Vintage
        is_vintage = res['Tipe_Pasokan'] == 'One-off Vintage'
        res.loc[is_vintage, 'Safety_Stock_Dinamis'] = 0
        res.loc[is_vintage, 'ROP_Dinamis'] = 0
        
        # Status Dinamis
        res['Gap_ke_ROP_Dinamis'] = res['Stok_Akhir'] - res['ROP_Dinamis']
        res['Status_Stok_Dinamis'] = np.where(
            is_vintage,
            'Sourcing per item',
            np.where(res['Gap_ke_ROP_Dinamis'] <= 0, 'PERLU RESTOCK', np.where(res['Gap_ke_ROP_Dinamis'] <= 2, 'Mendekati ROP', 'Aman'))
        )
        
        # 7. EOQ & MOQ Guardrails
        demand_ann = res['Demand_Mean_Harian'] * 365
        holding_ann = self.holding_cost_rate * res['Modal_Unit_Rp']
        res['EOQ_Unit'] = np.ceil(np.sqrt((2 * demand_ann * self.order_cost) / holding_ann)).astype(int)
        res['MOQ_Supplier'] = res['MOQ_Unit'].fillna(0).astype(int)
        
        res['Qty_Rekomendasi_PO'] = np.where(
            res['Status_Stok_Dinamis'] == 'PERLU RESTOCK',
            np.maximum(res['EOQ_Unit'], res['MOQ_Supplier']),
            0
        )
        res['Total_Modal_PO_Rp'] = res['Qty_Rekomendasi_PO'] * res['Modal_Unit_Rp']
        
        self.result_df = res
        return res

    def export_recommendations(self, output_path: str = 'rekomendasi_po_supplier.xlsx'):
        """Mengekspor laporan rekomendasi pemesanan ulang ke file Excel."""
        if self.result_df is None:
            self.run_pipeline()
            
        po_urgent = self.result_df[self.result_df['Qty_Rekomendasi_PO'] > 0].copy()
        po_urgent = po_urgent.merge(
            self.df_supp[['Supplier_ID', 'Nama_Pemasok', 'Kota', 'Rating_Internal_5']],
            on='Supplier_ID',
            how='left'
        )
        
        cols = [
            'SKU', 'Nama_Produk', 'Kategori', 'Supplier_ID', 'Nama_Pemasok', 'Kota', 'Rating_Internal_5',
            'Stok_Akhir', 'Safety_Stock_Dinamis', 'ROP_Dinamis', 'Gap_ke_ROP_Dinamis',
            'EOQ_Unit', 'MOQ_Supplier', 'Qty_Rekomendasi_PO', 'Modal_Unit_Rp', 'Total_Modal_PO_Rp'
        ]
        po_urgent = po_urgent[cols].sort_values(by='Total_Modal_PO_Rp', ascending=False)
        
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            po_urgent.to_excel(writer, sheet_name='Rekomendasi_PO_Urgent', index=False)
            self.result_df.to_excel(writer, sheet_name='Status_Semua_SKU', index=False)
            
        print(f"[OK] Rekomendasi PO tersimpan: {output_path}")
        print(f"     - Total SKU Restock: {len(po_urgent)} SKU")
        print(f"     - Total Qty PO     : {po_urgent['Qty_Rekomendasi_PO'].sum():,} unit")
        print(f"     - Total Modal PO   : Rp {po_urgent['Total_Modal_PO_Rp'].sum():,.0f}")
        return output_path

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Inventory Control & ROP Optimization Engine")
    parser.add_argument('--dataset', default='dataset_supply_chain_pakaian_90an_shopee.xlsx', help="Path file Excel dataset")
    parser.add_argument('--output', default='rekomendasi_po_supplier.xlsx', help="Path output rekomendasi Excel")
    args = parser.parse_args()
    
    engine = InventoryOptimizationEngine(args.dataset)
    engine.run_pipeline()
    engine.export_recommendations(args.output)
