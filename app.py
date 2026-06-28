import streamlit as st
import joblib
import pandas as pd
import re
import plotly.express as px
from datetime import datetime
from supabase import create_client, Client
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

# ============================================================
# KONFIGURASI HALAMAN
# ============================================================
st.set_page_config(
    page_title="Klasifikasi Status Pelanggan",
    page_icon="🏠",
    layout="wide"
)

# ============================================================
# CSS
# ============================================================
st.markdown("""
<style>
    .top-header {
        background: linear-gradient(135deg, #1e3a5f, #2563eb);
        padding: 16px 24px;
        border-radius: 10px;
        margin-bottom: 24px;
    }
    .top-header h2 { color: white; margin: 0; font-size: 20px; }
    .top-header p  { color: #a0bcd8; margin: 2px 0 0 0; font-size: 13px; }

    .badge-Ask       { background:#DBEAFE; color:#1D4ED8; padding:5px 14px; border-radius:20px; font-weight:700; }
    .badge-Follow-Up { background:#D1FAE5; color:#065F46; padding:5px 14px; border-radius:20px; font-weight:700; }
    .badge-Hold      { background:#FEF3C7; color:#92400E; padding:5px 14px; border-radius:20px; font-weight:700; }
    .badge-Closing   { background:#FCE7F3; color:#9D174D; padding:5px 14px; border-radius:20px; font-weight:700; }

    .hasil-box {
        background:#f8fafc; border:1px solid #e2e8f0;
        border-radius:10px; padding:16px 20px; margin-bottom:12px;
    }
    .rekomen-box {
        background:#fffbeb; border-left:4px solid #f59e0b;
        border-radius:6px; padding:12px 16px; margin-top:8px;
    }
    .metric-card {
        background:white; border:1px solid #e2e8f0;
        border-radius:10px; padding:16px; text-align:center;
    }
    .metric-card .num  { font-size:32px; font-weight:700; color:#1e3a5f; }
    .metric-card .lbl  { font-size:13px; color:#64748b; margin-top:4px; }

    .empty-state { text-align:center; padding:40px 20px; color:#94a3b8; }
    .empty-state .icon { font-size:48px; margin-bottom:12px; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# KONEKSI SUPABASE
# ============================================================
@st.cache_resource
def init_supabase() -> Client:
    url = st.secrets["supabase"]["url"]
    key = st.secrets["supabase"]["key"]
    return create_client(url, key)

try:
    supabase = init_supabase()
    db_ok = True
except Exception as e:
    db_ok = False
    st.error(f"❌ Gagal koneksi database: {e}")

# ============================================================
# LOAD MODEL
# ============================================================
@st.cache_resource
def load_model():
    try:
        model = joblib.load('model_svm.pkl')
        tfidf = joblib.load('tfidf_vectorizer.pkl')
        return model, tfidf, True
    except Exception as e:
        return None, None, False

@st.cache_resource
def load_preprocessor():
    stemmer  = StemmerFactory().create_stemmer()
    stopword = StopWordRemoverFactory().create_stop_word_remover()
    return stemmer, stopword

model, tfidf, model_loaded = load_model()

# ============================================================
# FUNGSI PREPROCESSING
# ============================================================
def preprocess(text, stemmer, stopword_remover):
    if not isinstance(text, str) or text.strip() == '':
        return ''
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = stopword_remover.remove(text)
    text = stemmer.stem(text)
    return text

# ============================================================
# FUNGSI DATABASE
# ============================================================
def simpan_riwayat(nama_cs, teks_input, teks_bersih, status, confidence, rekomendasi):
    try:
        supabase.table('riwayat_prediksi').insert({
            'nama_cs':     nama_cs,
            'teks_input':  teks_input,
            'teks_bersih': teks_bersih,
            'status':      status,
            'confidence':  round(confidence, 2),
            'rekomendasi': rekomendasi,
        }).execute()
        return True
    except Exception as e:
        st.warning(f"Gagal simpan riwayat: {e}")
        return False

def ambil_riwayat(limit=100):
    try:
        res = supabase.table('riwayat_prediksi')\
            .select('*')\
            .order('created_at', desc=True)\
            .limit(limit)\
            .execute()
        return pd.DataFrame(res.data) if res.data else pd.DataFrame()
    except:
        return pd.DataFrame()

def ambil_statistik():
    try:
        res = supabase.table('riwayat_prediksi').select('status').execute()
        if res.data:
            df = pd.DataFrame(res.data)
            return df['status'].value_counts().to_dict()
        return {}
    except:
        return {}

def ambil_users_cs():
    try:
        res = supabase.table('users_cs').select('nama').eq('aktif', True).execute()
        return [r['nama'] for r in res.data] if res.data else ['CS 1']
    except:
        return ['CS 1', 'CS 2', 'CS 3']

def hapus_riwayat_semua():
    try:
        supabase.table('riwayat_prediksi').delete().neq('id', 0).execute()
        return True
    except:
        return False

# ============================================================
# DATA REKOMENDASI
# ============================================================
REKOMENDASI = {
    'Ask': {
        'icon': '📘',
        'teks': 'Edukasi produk dan kirim katalog wallpaper terbaru. Berikan informasi detail tentang material, ukuran, dan harga.'
    },
    'Follow Up': {
        'icon': '🔔',
        'teks': 'Lakukan reminder follow-up dalam 1-2 hari. Tanyakan keputusan pelanggan dan tawarkan bantuan lebih lanjut.'
    },
    'Hold': {
        'icon': '⏸️',
        'teks': 'Monitoring berkala setiap minggu. Catat alasan penundaan dan tunggu waktu yang tepat untuk follow-up.'
    },
    'Closing': {
        'icon': '✅',
        'teks': 'Proses transaksi segera! Kirim invoice, konfirmasi pembayaran, dan atur jadwal pemasangan.'
    }
}

# ============================================================
# HEADER
# ============================================================
st.markdown("""
<div class="top-header">
    <h2>🏠 Dashboard Klasifikasi Status Pelanggan</h2>
    <p>Wallpaper Indonesia ID — Sistem Pendukung Keputusan Customer Service</p>
</div>
""", unsafe_allow_html=True)

# ============================================================
# CEK MODEL
# ============================================================
if not model_loaded:
    st.error("❌ **Model tidak ditemukan!** Pastikan `model_svm.pkl` dan `tfidf_vectorizer.pkl` ada di folder yang sama.")
    st.stop()

# ============================================================
# NAVIGASI
# ============================================================
tab_home, tab_riwayat, tab_statistik = st.tabs(["🏠 Home", "📋 Riwayat Prediksi", "📊 Statistik"])

stemmer, stopword_remover = load_preprocessor()
list_cs = ambil_users_cs()

# ============================================================
# TAB 1 - HOME (KLASIFIKASI)
# ============================================================
with tab_home:
    col_input, col_hasil = st.columns([1, 1], gap="large")

    with col_input:
        st.markdown("#### Input Interaksi Pelanggan")
        st.caption("Masukkan ringkasan percakapan WhatsApp dengan pelanggan")

        nama_cs = st.selectbox("Nama CS", list_cs)

        teks_input = st.text_area(
            "Teks Interaksi Pelanggan",
            placeholder="Contoh: Pelanggan tanya harga wallpaper untuk 3 kamar dan minta dikirim desain...",
            height=140
        )

        tombol = st.button("🔍 Prediksi Status Pelanggan", use_container_width=True, type="primary")

    with col_hasil:
        st.markdown("#### Hasil Klasifikasi")

        if tombol:
            if not teks_input.strip():
                st.warning("⚠️ Teks interaksi tidak boleh kosong!")
            else:
                with st.spinner("Memproses..."):
                    teks_bersih = preprocess(teks_input, stemmer, stopword_remover)
                    vektor      = tfidf.transform([teks_bersih])
                    prediksi    = model.predict(vektor)[0]
                    confidence  = model.predict_proba(vektor).max() * 100
                    rekomen     = REKOMENDASI.get(prediksi, {})

                # Badge status
                badge_key = prediksi.replace(' ', '-')
                st.markdown(f"""
                <div class="hasil-box">
                    <div style="font-size:13px;color:#64748b;margin-bottom:6px;">Status Pelanggan</div>
                    <span class="badge-{badge_key}">{prediksi}</span>
                    &nbsp;
                    <span style="color:#64748b;font-size:13px;">Confidence: <strong>{confidence:.1f}%</strong></span>
                    <div style="background:#e2e8f0;border-radius:10px;height:8px;margin-top:10px;">
                        <div style="width:{confidence:.0f}%;height:8px;border-radius:10px;background:#2563eb;"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Rekomendasi
                st.markdown(f"""
                <div class="rekomen-box">
                    <div style="font-size:13px;font-weight:700;color:#92400E;margin-bottom:4px;">
                        {rekomen.get('icon','')} Rekomendasi Tindak Lanjut
                    </div>
                    <div style="font-size:14px;">{rekomen.get('teks','')}</div>
                </div>
                """, unsafe_allow_html=True)

                # Simpan ke database
                if db_ok:
                    simpan_riwayat(
                        nama_cs    = nama_cs,
                        teks_input = teks_input,
                        teks_bersih= teks_bersih,
                        status     = prediksi,
                        confidence = confidence,
                        rekomendasi= rekomen.get('teks','')
                    )
                    st.success("✅ Hasil tersimpan ke database")
        else:
            st.markdown("""
            <div class="empty-state">
                <div class="icon">📄</div>
                <p>Masukkan teks percakapan pelanggan<br>untuk mendapatkan prediksi status</p>
            </div>
            """, unsafe_allow_html=True)

# ============================================================
# TAB 2 - RIWAYAT PREDIKSI
# ============================================================
with tab_riwayat:
    st.markdown("#### Riwayat Prediksi")
    st.caption("Log semua aktivitas klasifikasi status pelanggan")

    col_r1, col_r2 = st.columns([3, 1])
    with col_r1:
        filter_status = st.selectbox(
            "Filter Status",
            ["Semua", "Ask", "Follow Up", "Hold", "Closing"]
        )
    with col_r2:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🔄 Refresh", use_container_width=True):
            st.rerun()

    df_riwayat = ambil_riwayat(limit=200)

    if df_riwayat.empty:
        st.markdown("""
        <div class="empty-state">
            <div class="icon">📋</div>
            <p>Belum ada riwayat prediksi.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Filter
        if filter_status != "Semua":
            df_riwayat = df_riwayat[df_riwayat['status'] == filter_status]

        # Format tampilan
        df_tampil = df_riwayat[['nama_cs', 'teks_input', 'status', 'confidence', 'rekomendasi', 'created_at']].copy()
        df_tampil.columns = ['CS', 'Teks Input', 'Status', 'Confidence (%)', 'Rekomendasi', 'Waktu']
        df_tampil['Confidence (%)'] = df_tampil['Confidence (%)'].apply(lambda x: f"{x:.1f}%")
        df_tampil['Waktu'] = pd.to_datetime(df_tampil['Waktu']).dt.strftime('%d/%m/%Y %H:%M')

        st.dataframe(df_tampil, use_container_width=True, hide_index=True)

        # Download
        csv = df_tampil.to_csv(index=False).encode('utf-8')
        st.download_button(
            "⬇️ Download CSV",
            data=csv,
            file_name=f"riwayat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime='text/csv'
        )

        # Hapus semua
        with st.expander("⚠️ Hapus Semua Riwayat"):
            st.warning("Tindakan ini tidak dapat dibatalkan!")
            if st.button("🗑️ Hapus Semua", type="secondary"):
                if hapus_riwayat_semua():
                    st.success("Riwayat berhasil dihapus!")
                    st.rerun()

# ============================================================
# TAB 3 - STATISTIK
# ============================================================
with tab_statistik:
    st.markdown("#### Statistik Klasifikasi")

    statistik = ambil_statistik()
    df_riwayat_all = ambil_riwayat(limit=1000)

    if not statistik:
        st.markdown("""
        <div class="empty-state">
            <div class="icon">📊</div>
            <p>Belum ada data untuk ditampilkan.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Metric cards
        total = sum(statistik.values())
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Total Prediksi",  total)
        c2.metric("Ask",       statistik.get('Ask', 0))
        c3.metric("Follow Up", statistik.get('Follow Up', 0))
        c4.metric("Hold",      statistik.get('Hold', 0))
        c5.metric("Closing",   statistik.get('Closing', 0))

        st.markdown("---")

        col_g1, col_g2 = st.columns(2)

        # Pie chart distribusi
        with col_g1:
            st.markdown("**Distribusi Status Pelanggan**")
            df_pie = pd.DataFrame({
                'Status': list(statistik.keys()),
                'Jumlah': list(statistik.values())
            })
            colors = {
                'Ask':       '#DBEAFE',
                'Follow Up': '#D1FAE5',
                'Hold':      '#FEF3C7',
                'Closing':   '#FCE7F3'
            }
            fig_pie = px.pie(
                df_pie, values='Jumlah', names='Status',
                color='Status',
                color_discrete_map=colors,
                hole=0.4
            )
            fig_pie.update_layout(margin=dict(t=10, b=10))
            st.plotly_chart(fig_pie, use_container_width=True)

        # Bar chart per status
        with col_g2:
            st.markdown("**Jumlah per Status**")
            fig_bar = px.bar(
                df_pie, x='Status', y='Jumlah',
                color='Status',
                color_discrete_map=colors,
                text='Jumlah'
            )
            fig_bar.update_traces(textposition='outside')
            fig_bar.update_layout(
                showlegend=False,
                margin=dict(t=10, b=10),
                yaxis_title="Jumlah Prediksi"
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        # Tren waktu jika ada data
        if not df_riwayat_all.empty and 'created_at' in df_riwayat_all.columns:
            st.markdown("**Tren Prediksi per Hari**")
            df_tren = df_riwayat_all.copy()
            df_tren['tanggal'] = pd.to_datetime(df_tren['created_at']).dt.date
            df_tren_group = df_tren.groupby(['tanggal', 'status']).size().reset_index(name='jumlah')

            fig_line = px.line(
                df_tren_group, x='tanggal', y='jumlah',
                color='status',
                color_discrete_map={
                    'Ask': '#3B82F6', 'Follow Up': '#10B981',
                    'Hold': '#F59E0B', 'Closing': '#EC4899'
                },
                markers=True
            )
            fig_line.update_layout(
                xaxis_title="Tanggal",
                yaxis_title="Jumlah Prediksi",
                margin=dict(t=10, b=10)
            )
            st.plotly_chart(fig_line, use_container_width=True)
