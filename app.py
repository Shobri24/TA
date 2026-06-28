import streamlit as st
import joblib
import pandas as pd
import re
import html
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
        display:flex; align-items:center; gap:12px;
        padding: 14px 0 18px 0;
        border-bottom: 1px solid #e2e8f0;
        margin-bottom: 20px;
    }
    .top-header .logo {
        width:40px; height:40px; border-radius:10px;
        background: linear-gradient(135deg, #1e3a5f, #2563eb);
        display:flex; align-items:center; justify-content:center;
        font-size:20px; color:white;
    }
    .top-header h2 { color:#0f172a; margin:0; font-size:18px; font-weight:700; }
    .top-header p  { color:#64748b; margin:0; font-size:12px; }

    /* Badge status - bentuk pill kecil dengan teks status */
    .badge-Ask       { background:#DBEAFE; color:#1D4ED8; padding:4px 12px; border-radius:20px; font-weight:600; font-size:13px; }
    .badge-Follow-Up { background:#FFEDD5; color:#C2410C; padding:4px 12px; border-radius:20px; font-weight:600; font-size:13px; }
    .badge-Hold      { background:#EDE9FE; color:#6D28D9; padding:4px 12px; border-radius:20px; font-weight:600; font-size:13px; }
    .badge-Closing   { background:#D1FAE5; color:#047857; padding:4px 12px; border-radius:20px; font-weight:600; font-size:13px; }

    /* Icon bulat untuk hasil klasifikasi (sesuai gambar) */
    .icon-circle {
        width:38px; height:38px; border-radius:50%;
        display:flex; align-items:center; justify-content:center;
        font-size:18px; flex-shrink:0;
    }
    .icon-Ask       { background:#DBEAFE; color:#1D4ED8; }
    .icon-Follow-Up { background:#FFEDD5; color:#C2410C; }
    .icon-Hold      { background:#EDE9FE; color:#6D28D9; }
    .icon-Closing   { background:#D1FAE5; color:#047857; }

    .card-box {
        background:white; border:1px solid #e2e8f0;
        border-radius:10px; padding:20px 24px; margin-bottom:16px;
    }
    .hasil-row { display:flex; align-items:center; gap:14px; }
    .hasil-row .status-name { font-size:18px; font-weight:700; color:#0f172a; }
    .confidence-chip {
        background:#f1f5f9; color:#475569; font-size:11px; font-weight:600;
        padding:3px 10px; border-radius:6px; margin-left:6px;
    }
    .progress-bar-bg { background:#e2e8f0; border-radius:10px; height:7px; margin-top:14px; }
    .progress-bar-Ask       { background:#2563eb; }
    .progress-bar-Follow-Up { background:#ea580c; }
    .progress-bar-Hold      { background:#7c3aed; }
    .progress-bar-Closing   { background:#059669; }

    .rekomen-box {
        background:#eff6ff; border-left:4px solid #2563eb;
        border-radius:6px; padding:14px 18px; margin-top:14px;
    }
    .rekomen-box .judul { font-size:13px; font-weight:700; color:#1e40af; margin-bottom:4px; }
    .rekomen-box .isi { font-size:14px; color:#334155; }

    .metric-card {
        background:white; border:1px solid #e2e8f0;
        border-radius:10px; padding:16px; text-align:center;
    }
    .metric-card .num  { font-size:32px; font-weight:700; color:#1e3a5f; }
    .metric-card .lbl  { font-size:13px; color:#64748b; margin-top:4px; }

    .empty-state { text-align:center; padding:40px 20px; color:#94a3b8; }
    .empty-state .icon { font-size:48px; margin-bottom:12px; }

    /* Tabel riwayat custom */
    .riwayat-table-wrap { overflow-x:auto; }
    .riwayat-table { width:100%; border-collapse:collapse; font-size:13px; }
    .riwayat-table th {
        text-align:left; padding:10px 12px; color:#64748b;
        font-weight:600; border-bottom:2px solid #e2e8f0; white-space:nowrap;
    }
    .riwayat-table td {
        padding:12px; border-bottom:1px solid #f1f5f9; vertical-align:top; color:#334155;
    }
    .riwayat-table tr:hover td { background:#f8fafc; }
    .conf-pill {
        background:#f1f5f9; color:#334155; padding:2px 8px;
        border-radius:6px; font-size:12px; font-weight:600;
    }
    .col-teks { max-width:220px; }
    .col-rekom { max-width:260px; color:#64748b; }
    .col-waktu { white-space:nowrap; color:#94a3b8; font-size:12px; }
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
def simpan_riwayat(teks_input, teks_bersih, status, confidence, rekomendasi, nama_cs="CS"):
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

def hitung_total_riwayat():
    try:
        res = supabase.table('riwayat_prediksi').select('id', count='exact').execute()
        return res.count if res.count is not None else 0
    except:
        return 0

def hapus_riwayat_semua():
    try:
        supabase.table('riwayat_prediksi').delete().neq('id', 0).execute()
        return True
    except:
        return False

# ============================================================
# DATA REKOMENDASI & ICON
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
        'icon': '⏳',
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
    <div class="logo">🏠</div>
    <div>
        <h2>Dashboard Klasifikasi Status Pelanggan</h2>
        <p>Wallpaper Indonesia ID</p>
    </div>
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
total_riwayat_count = hitung_total_riwayat() if db_ok else 0

label_home     = "🏠 Home"
label_riwayat  = f"🕘 Riwayat ({total_riwayat_count})" if total_riwayat_count > 0 else "🕘 Riwayat"
label_statistik = "📊 Statistik"

tab_home, tab_riwayat, tab_statistik = st.tabs([label_home, label_riwayat, label_statistik])

stemmer, stopword_remover = load_preprocessor()

# ============================================================
# TAB 1 - HOME (KLASIFIKASI) — Layout 1 kolom sesuai gambar
# ============================================================
with tab_home:
    st.markdown("#### Input Interaksi Pelanggan")
    st.caption("Masukkan ringkasan percakapan WhatsApp dengan pelanggan untuk mengklasifikasikan status mereka")

    teks_input = st.text_area(
        "Teks Interaksi Pelanggan",
        placeholder="Contoh: Pelanggan tanya harga wallpaper untuk 3 kamar dan minta dikabari besok...",
        height=100,
        label_visibility="collapsed"
    )

    tombol = st.button("🔍 Prediksi Status Pelanggan", use_container_width=True, type="primary")

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

            badge_key = prediksi.replace(' ', '-')

            hasil_html = (
                f'<div class="card-box"><div class="hasil-row">'
                f'<div class="icon-circle icon-{badge_key}">{rekomen.get("icon","")}</div>'
                f'<div><span class="status-name">{prediksi}</span>'
                f'<span class="confidence-chip">Confidence: {confidence:.1f}%</span></div></div>'
                f'<div class="progress-bar-bg"><div style="width:{confidence:.0f}%;height:7px;border-radius:10px;" '
                f'class="progress-bar-{badge_key}"></div></div>'
                f'<div class="rekomen-box"><div class="judul">{rekomen.get("icon","")} Rekomendasi Tindak Lanjut</div>'
                f'<div class="isi">{rekomen.get("teks","")}</div></div></div>'
            )
            st.markdown(hasil_html, unsafe_allow_html=True)

            if db_ok:
                simpan_riwayat(
                    teks_input  = teks_input,
                    teks_bersih = teks_bersih,
                    status      = prediksi,
                    confidence  = confidence,
                    rekomendasi = rekomen.get('teks',''),
                    nama_cs     = "CS"
                )
                st.success("✅ Hasil tersimpan ke database")
    else:
        st.markdown("""
        <div class="card-box">
            <div class="empty-state">
                <div class="icon">📄</div>
                <p>Masukkan teks percakapan pelanggan di atas<br>untuk mendapatkan prediksi status dan rekomendasi tindak lanjut</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

# ============================================================
# TAB 2 - RIWAYAT PREDIKSI — Tabel custom sesuai gambar
# ============================================================
with tab_riwayat:
    st.markdown("#### Riwayat Prediksi")
    st.caption("Log data klasifikasi status pelanggan yang telah dilakukan")

    col_r1, col_r2 = st.columns([3, 1])
    with col_r1:
        filter_status = st.selectbox(
            "Filter Status",
            ["Semua", "Ask", "Follow Up", "Hold", "Closing"],
            label_visibility="collapsed"
        )
    with col_r2:
        if st.button("🔄 Refresh", use_container_width=True):
            st.rerun()

    df_riwayat = ambil_riwayat(limit=200)

    if df_riwayat.empty:
        st.markdown("""
        <div class="card-box">
            <div class="empty-state">
                <div class="icon">📋</div>
                <p>Belum ada riwayat prediksi.</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        if filter_status != "Semua":
            df_riwayat = df_riwayat[df_riwayat['status'] == filter_status]

        df_riwayat = df_riwayat.reset_index(drop=True)
        df_riwayat['no_urut'] = range(len(df_riwayat), 0, -1)
        df_riwayat['waktu_fmt'] = pd.to_datetime(df_riwayat['created_at']).dt.strftime('%-d/%-m/%Y, %H.%M.%S')

        def potong(teks, n=80):
            teks = html.escape(str(teks))
            return teks if len(teks) <= n else teks[:n].rstrip() + "..."

        baris_html = ""
        for _, row in df_riwayat.iterrows():
            badge_key = str(row['status']).replace(' ', '-')
            status_aman = html.escape(str(row['status']))
            icon = REKOMENDASI.get(row['status'], {}).get('icon', '')
            baris_html += (
                "<tr>"
                f"<td>{row['no_urut']}</td>"
                f"<td class=\"col-teks\">{potong(row['teks_input'])}</td>"
                f"<td><span class=\"badge-{badge_key}\">{icon} {status_aman}</span></td>"
                f"<td><span class=\"conf-pill\">{row['confidence']:.1f}%</span></td>"
                f"<td class=\"col-rekom\">{potong(row['rekomendasi'], 90)}</td>"
                f"<td class=\"col-waktu\">{row['waktu_fmt']}</td>"
                "</tr>"
            )

        tabel_html = (
            '<div class="card-box"><div class="riwayat-table-wrap">'
            '<table class="riwayat-table"><thead><tr>'
            '<th>No</th><th>Teks Input</th><th>Status</th>'
            '<th>Confidence</th><th>Rekomendasi</th><th>Timestamp</th>'
            f'</tr></thead><tbody>{baris_html}</tbody></table>'
            '</div></div>'
        )
        st.markdown(tabel_html, unsafe_allow_html=True)

        # Download
        df_export = df_riwayat[['nama_cs', 'teks_input', 'status', 'confidence', 'rekomendasi', 'created_at']].copy()
        df_export.columns = ['CS', 'Teks Input', 'Status', 'Confidence (%)', 'Rekomendasi', 'Waktu']
        csv = df_export.to_csv(index=False).encode('utf-8')
        st.download_button(
            "⬇️ Download CSV",
            data=csv,
            file_name=f"riwayat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime='text/csv'
        )

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
        total = sum(statistik.values())
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Total Prediksi",  total)
        c2.metric("Ask",       statistik.get('Ask', 0))
        c3.metric("Follow Up", statistik.get('Follow Up', 0))
        c4.metric("Hold",      statistik.get('Hold', 0))
        c5.metric("Closing",   statistik.get('Closing', 0))

        st.markdown("---")

        col_g1, col_g2 = st.columns(2)

        colors = {
            'Ask':       '#DBEAFE',
            'Follow Up': '#FFEDD5',
            'Hold':      '#EDE9FE',
            'Closing':   '#D1FAE5'
        }

        with col_g1:
            st.markdown("**Distribusi Status Pelanggan**")
            df_pie = pd.DataFrame({
                'Status': list(statistik.keys()),
                'Jumlah': list(statistik.values())
            })
            fig_pie = px.pie(
                df_pie, values='Jumlah', names='Status',
                color='Status',
                color_discrete_map=colors,
                hole=0.4
            )
            fig_pie.update_layout(margin=dict(t=10, b=10))
            st.plotly_chart(fig_pie, use_container_width=True)

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

        if not df_riwayat_all.empty and 'created_at' in df_riwayat_all.columns:
            st.markdown("**Tren Prediksi per Hari**")
            df_tren = df_riwayat_all.copy()
            df_tren['tanggal'] = pd.to_datetime(df_tren['created_at']).dt.date
            df_tren_group = df_tren.groupby(['tanggal', 'status']).size().reset_index(name='jumlah')

            fig_line = px.line(
                df_tren_group, x='tanggal', y='jumlah',
                color='status',
                color_discrete_map={
                    'Ask': '#2563eb', 'Follow Up': '#ea580c',
                    'Hold': '#7c3aed', 'Closing': '#059669'
                },
                markers=True
            )
            fig_line.update_layout(
                xaxis_title="Tanggal",
                yaxis_title="Jumlah Prediksi",
                margin=dict(t=10, b=10)
            )
            st.plotly_chart(fig_line, use_container_width=True)