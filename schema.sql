-- ============================================================
-- SCHEMA DATABASE - Jalankan di Supabase SQL Editor
-- ============================================================

-- 1. Tabel riwayat prediksi CS
CREATE TABLE IF NOT EXISTS riwayat_prediksi (
    id          BIGSERIAL PRIMARY KEY,
    nama_cs     TEXT DEFAULT 'Customer Service',
    teks_input  TEXT NOT NULL,
    teks_bersih TEXT,
    status      TEXT NOT NULL CHECK (status IN ('Ask', 'Follow Up', 'Hold', 'Closing')),
    confidence  FLOAT NOT NULL,
    rekomendasi TEXT,
    created_at  TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- 2. Tabel data pelanggan
CREATE TABLE IF NOT EXISTS data_pelanggan (
    id          BIGSERIAL PRIMARY KEY,
    nama        TEXT NOT NULL,
    nomor_hp    TEXT,
    teks        TEXT NOT NULL,
    label       TEXT CHECK (label IN ('Ask', 'Follow Up', 'Hold', 'Closing')),
    bulan       TEXT,
    created_at  TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- 3. Tabel users CS
CREATE TABLE IF NOT EXISTS users_cs (
    id          BIGSERIAL PRIMARY KEY,
    nama        TEXT NOT NULL,
    divisi      TEXT DEFAULT 'Customer Service',
    aktif       BOOLEAN DEFAULT TRUE,
    created_at  TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Insert user CS default
INSERT INTO users_cs (nama, divisi) VALUES
    ('CS 1', 'Customer Service'),
    ('CS 2', 'Customer Service'),
    ('CS 3', 'Customer Service')
ON CONFLICT DO NOTHING;

-- Index untuk performa query
CREATE INDEX IF NOT EXISTS idx_riwayat_status     ON riwayat_prediksi(status);
CREATE INDEX IF NOT EXISTS idx_riwayat_created_at ON riwayat_prediksi(created_at);
CREATE INDEX IF NOT EXISTS idx_pelanggan_label    ON data_pelanggan(label);
