# coding: utf-8
import pandas as pd
import numpy as np
import sqlite3
import traceback
import warnings
from flask import Flask, render_template_string, request, jsonify
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

warnings.filterwarnings('ignore')

# ================= 設定區 (請確認路徑與表名) =================
DB_MARKET = r"D:\python\py\股票資料.db"
DB_FINANCE = r"D:\python\py\財務報表.db"

PERIODS = [1, 2, 3, 5]
TABLE_PRICE = '每日收盤行情'
TABLE_MARGIN = '融資融劵彙總'
TABLE_INST = '三大法人買賣超日報'
# =============================================================

app = Flask(__name__)

def get_db_connection(db_path):
    return sqlite3.connect(db_path)

def fix_tw_date(date_str):
    s = str(date_str).strip().replace('/', '').replace('-', '')
    if s.lower() in ['nan', 'none', '']: return np.nan
    try:
        if len(s) == 7: return str(int(s[:3]) + 1911) + s[3:]
        elif len(s) == 6: return str(int(s[:2]) + 1911) + s[2:]
        return s
    except:
        return s

def safe_pct_change(curr, prev):
    return ((curr - prev) / prev * 100) if prev and prev != 0 else 0

# ==========================================
# 模組一：表格流的 SQL 極速掃描引擎 (統一單位：股)
# ==========================================
def load_and_analyze(conn, start_date, end_date, criteria):
    m_th = float(criteria.get('margin_th', -2))
    s_th = float(criteria.get('short_th', 2))
    min_p = float(criteria.get('min_price_change', 0))
    min_i_shares = int(criteria.get('min_inst_buy', 100)) * 1000 
    mode = criteria.get('strategy_mode', 'all')

    # 🌟 經過照妖鏡確認，日期為平整格式，且欄位名稱為「今日餘額」，使用最乾淨高效的語法
    sql = f"""
    SELECT 
        A.證券代號, A.證券名稱, P_End.收盤價 AS 收盤_End, P_Start.收盤價 AS 收盤_Start,
        IFNULL(M_End.融券今日餘額, 0) AS 券餘_End, IFNULL(M_Start.融券今日餘額, 0) AS 券餘_Start,
        IFNULL(M_End.融資今日餘額, 0) AS 資餘_End, IFNULL(M_Start.融資今日餘額, 0) AS 資餘_Start,
        IFNULL(Inst.法人合計, 0) AS 法人合計, IFNULL(Inst.外資, 0) AS 外資, IFNULL(Inst.投信, 0) AS 投信
    FROM (SELECT DISTINCT 證券代號, 證券名稱 FROM {TABLE_PRICE}) A
    JOIN {TABLE_PRICE} P_End ON A.證券代號 = P_End.證券代號 AND P_End.日期 = '{end_date}'
    LEFT JOIN {TABLE_PRICE} P_Start ON A.證券代號 = P_Start.證券代號 AND P_Start.日期 = '{start_date}'
    LEFT JOIN {TABLE_MARGIN} M_End ON A.證券代號 = M_End.證券代號 AND M_End.日期 = '{end_date}'
    LEFT JOIN {TABLE_MARGIN} M_Start ON A.證券代號 = M_Start.證券代號 AND M_Start.日期 = '{start_date}'
    LEFT JOIN (
        SELECT 證券代號, 
               SUM(三大法人買賣超股數) as 法人合計, 
               SUM(外陸資買賣超股數_不含外資自營商) as 外資, 
               SUM(投信買賣超股數) as 投信
        FROM {TABLE_INST} WHERE 日期 > '{start_date}' AND 日期 <= '{end_date}' GROUP BY 證券代號
    ) Inst ON A.證券代號 = Inst.證券代號
    """
    df = pd.read_sql(sql, conn)
    for c in ['收盤_End', '收盤_Start', '券餘_End', '券餘_Start', '資餘_End', '資餘_Start']:
        df[c] = pd.to_numeric(df[c].astype(str).str.replace(',', ''), errors='coerce').fillna(0)

    df['股價漲跌幅%'] = df.apply(lambda x: safe_pct_change(x['收盤_End'], x['收盤_Start']), axis=1)
    df['融資增減幅%'] = df.apply(lambda x: safe_pct_change(x['資餘_End'], x['資餘_Start']), axis=1)
    df['融券增減幅%'] = df.apply(lambda x: safe_pct_change(x['券餘_End'], x['券餘_Start']), axis=1)
    df['法人區間買賣'] = df['法人合計']

    def get_tags(row):
        t = []
        is_sed = row['融資增減幅%'] <= m_th
        is_sqz = row['融券增減幅%'] >= s_th
        if is_sed: t.append('<span class="tag-box tag-sediment">★ 籌碼沉澱 (資減)</span>')
        if is_sqz: t.append('<span class="tag-box tag-squeeze">🔥 軋空蓄勢 (券增)</span>')
        if is_sed and is_sqz: t.insert(0, '<span class="tag-box tag-strict">🎯 雙重優勢</span>')
        if mode == 'strict_and' and not (is_sed and is_sqz): return None
        return "".join(t) if t else None

    df['籌碼預測_HTML'] = df.apply(get_tags, axis=1)
    df = df[df['籌碼預測_HTML'].notnull()]
    
    # 用「股數」過濾
    df = df[(df['股價漲跌幅%'] >= min_p) & (df['法人區間買賣'] >= min_i_shares)]
    
    # 將股數加上千位符號，前端直接顯示
    df['法人區間買賣_顯示'] = df['法人區間買賣'].apply(lambda x: f"{int(x):,}")
    df['備註分析'] = df.apply(lambda r: f"外資買 {int(r['外資']):,} 股" if r['外資']>0 else "", axis=1)

    return df.rename(columns={'收盤_End': '收盤價'}).to_dict('records')

# ==========================================
# 模組二：AI 流的特徵工程與模型引擎
# ==========================================
def get_single_stock_data(stock_id, target_date_str=None):
    conn = sqlite3.connect(DB_MARKET)
    try:
        df_margin = pd.read_sql_query(f"SELECT * FROM {TABLE_MARGIN} WHERE 證券代號 = '{stock_id}'", conn)
        df_inst = pd.read_sql_query(f"SELECT * FROM {TABLE_INST} WHERE 證券代號 = '{stock_id}'", conn)
        df_price = pd.read_sql_query(f"SELECT 日期, 證券代號, 成交股數, 收盤價 FROM {TABLE_PRICE} WHERE 證券代號 = '{stock_id}'", conn)
        if df_price.empty: return pd.DataFrame()
        
        for df in [df_margin, df_inst, df_price]:
            df['日期'] = pd.to_datetime(df['日期'].apply(fix_tw_date), errors='coerce')
            df.dropna(subset=['日期'], inplace=True)
            if '證券代號' in df.columns: df['證券代號'] = df['證券代號'].astype(str).str.strip()
            
        df_price['收盤價'] = pd.to_numeric(df_price['收盤價'].astype(str).str.replace(',', ''), errors='coerce')
        df_price['成交股數'] = pd.to_numeric(df_price['成交股數'].astype(str).str.replace(',', ''), errors='coerce')
        df = pd.merge(df_price, df_margin, on=['日期', '證券代號'], how='left')
        df = pd.merge(df, df_inst, on=['日期', '證券代號'], how='left')
        
        if target_date_str:
            target_dt = pd.to_datetime(fix_tw_date(target_date_str), errors='coerce')
            if pd.notnull(target_dt):
                df = df[df['日期'] <= target_dt]
                
        return df.sort_values('日期').reset_index(drop=True)
    finally: conn.close()

def get_fundamental_data(stock_id):
    conn = sqlite3.connect(DB_FINANCE)
    try:
        tables_df = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table';", conn)
        income_tables = [name for name in tables_df['name'] if name.startswith('綜合損益表_')]
        for table in reversed(income_tables):
            df = pd.read_sql_query(f"SELECT * FROM `{table}` WHERE 公司代號 = '{stock_id}'", conn)
            if not df.empty: return df 
        return pd.DataFrame()
    finally: conn.close()

def process_features(df):
    if df.empty: return None
    # 🌟 已將「今日餘額」加入 AI 模組的優先讀取名單
    margin_col = '融資今日餘額' if '融資今日餘額' in df.columns else ('融資當日餘額' if '融資當日餘額' in df.columns else '融資(張)_餘額')
    short_col = '融券今日餘額' if '融券今日餘額' in df.columns else ('融券當日餘額' if '融券當日餘額' in df.columns else '融券(張)_餘額')
    inst_col = '三大法人買賣超股數' if '三大法人買賣超股數' in df.columns else ('外陸資買賣超股數_不含外資自營商' if '外陸資買賣超股數_不含外資自營商' in df.columns else '合計')
    
    cols = [margin_col, short_col, inst_col, '收盤價', '成交股數']
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', ''), errors='coerce').ffill().fillna(0)
            
    df['股價日增減(%)'] = df['收盤價'].pct_change() * 100
    df['三大法人買賣超(與前日差)'] = df[inst_col].diff()
    df['融資日增減(%)'] = df[margin_col].pct_change() * 100
    df['融券日增減(%)'] = df[short_col].pct_change() * 100
    df['資券比(%)'] = (df[short_col] / df[margin_col].replace(0, np.nan)) * 100
    df['20MA'] = df['收盤價'].rolling(window=20).mean().bfill()
    df['估算維持率(%)'] = (df['收盤價'] / df['20MA'].replace(0, np.nan)) * 166
    df['法人成交佔比(%)'] = (df[inst_col] / df['成交股數'].replace(0, np.nan)) * 100
    
    df.replace([np.inf, -np.inf], 0, inplace=True)
    df['明日收盤價'] = df['收盤價'].shift(-1)
    df['明日漲跌_Y'] = np.where(df['明日收盤價'] > df['收盤價'], 1, 0)
    
    def get_margin_alert(row):
        alerts = []
        if row.get('資券比(%)', 0) > 30 and row.get('股價日增減(%)', 0) > 2: alerts.append("🔥軋空潛力")
        if row.get(inst_col, 0) < 0 and row.get('融資日增減(%)', 0) > 1 and row.get('股價日增減(%)', 0) < 0: alerts.append("🚨散戶接刀")
        if row.get(inst_col, 0) > 0 and row.get('融資日增減(%)', 0) < -1: alerts.append("💎籌碼沉澱")
        return " ".join(alerts) if alerts else ""
    df['籌碼警示'] = df.apply(get_margin_alert, axis=1)
    df.rename(columns={margin_col: '融資當日餘額', short_col: '融券當日餘額', inst_col: '三大法人買賣超股數'}, inplace=True)
    return df

# ==========================================
# Flask API 路由配置
# ==========================================
@app.route('/', methods=['GET', 'POST'])
def index():
    # 預設基準日期為我們剛剛發現三張表都有資料的最後一天
    target_date = request.form.get('target_date', '20251023') 
    criteria = {
        'min_price_change': request.form.get('min_price_change', 0),
        'min_inst_buy': request.form.get('min_inst_buy', 100),
        'margin_th': request.form.get('margin_th', -2),
        'short_th': request.form.get('short_th', 2),
        'strategy_mode': request.form.get('strategy_mode', 'all')
    }
    results = {}
    searched = False # 🌟 新增：用來記錄是否有按下搜尋按鈕
    
    if request.method == 'POST':
        searched = True
        conn = get_db_connection(DB_MARKET)
        all_dates = sorted(pd.read_sql(f"SELECT DISTINCT 日期 FROM {TABLE_PRICE} WHERE 日期<='{target_date}' ORDER BY 日期 DESC LIMIT 10", conn)['日期'].tolist())
        if all_dates:
            for p in PERIODS:
                if len(all_dates) > p:
                    results[p] = load_and_analyze(conn, all_dates[-(p+1)], all_dates[-1], criteria)
        conn.close()
    return render_template_string(HTML_TEMPLATE, results=results, target_date=target_date, criteria=criteria, searched=searched)

@app.route('/api/predict', methods=['POST'])
def api_predict():
    try:
        stock_id = str(request.json.get('stock_id')).strip()
        target_date = str(request.json.get('target_date')).strip() 
        
        df_raw = get_single_stock_data(stock_id, target_date)
        df_fund = get_fundamental_data(stock_id)
        
        if df_raw.empty: return jsonify({'error': '市場資料不足以進行 AI 運算'})
        
        df_processed = process_features(df_raw)
        features = ['收盤價', '股價日增減(%)', '三大法人買賣超股數', '三大法人買賣超(與前日差)', '法人成交佔比(%)', '融資當日餘額', '融資日增減(%)', '融券當日餘額', '融券日增減(%)', '資券比(%)', '估算維持率(%)']
        df_ml = df_processed.dropna(subset=features + ['明日漲跌_Y'])
        
        res = {'accuracy': 0, 'train_count': int(len(df_ml)), 'prob_up': 0.0, 'prob_down': 0.0, 'recent_data': [], 'fund_data': [], 'feature_analysis': []}
        
        if len(df_ml) >= 20:
            X, y = df_ml[features], df_ml['明日漲跌_Y']
            split_idx = int(len(df_ml) * 0.8)
            model = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
            model.fit(X.iloc[:split_idx], y.iloc[:split_idx])
            
            y_pred = model.predict(X.iloc[split_idx:])
            res['accuracy'] = float(round(accuracy_score(y.iloc[split_idx:], y_pred) * 100, 2))
            
            latest_row = df_ml.iloc[-1]
            if latest_row.get('三大法人買賣超股數', 0) < 0 and latest_row.get('融資日增減(%)', 0) > 1:
                res['prob_up'] = 0.0; res['prob_down'] = 100.0
            else:
                prob = model.predict_proba(X.iloc[-1:])[0]
                classes = list(model.classes_)
                res['prob_up'] = float(round(prob[classes.index(1)] * 100 if 1 in classes else 0.0, 2))
                res['prob_down'] = float(round(prob[classes.index(0)] * 100 if 0 in classes else 0.0, 2))
            
            importances = model.feature_importances_
            for i, col in enumerate(features):
                raw_val = X.iloc[-1:][col].values[0]
                val = raw_val.item() if hasattr(raw_val, 'item') else raw_val
                res['feature_analysis'].append({'feature': col, 'value': round(val, 2) if isinstance(val, float) else val, 'weight': round(float(importances[i] * 100), 2)})
            res['feature_analysis'].sort(key=lambda x: x['weight'], reverse=True)
            
            recent = df_processed[['日期', '收盤價', '股價日增減(%)', '三大法人買賣超股數', '融資日增減(%)', '資券比(%)', '估算維持率(%)', '籌碼警示']].tail(4).copy()
            recent['日期'] = recent['日期'].dt.strftime('%Y-%m-%d')
            res['recent_data'] = recent.fillna(0).to_dict(orient='records')
            
        if not df_fund.empty:
            yr_col = next((c for c in df_fund.columns if '年' in c), '年度')
            q_col = next((c for c in df_fund.columns if '季' in c), '季別')
            eps_col = next((c for c in df_fund.columns if '盈餘' in c or 'EPS' in c.upper()), None)
            rev_col = next((c for c in df_fund.columns if '收入' in c), None)
            cols = [c for c in [yr_col, q_col, eps_col, rev_col] if c and c in df_fund.columns]
            if cols:
                temp_fund = df_fund[cols].tail(2).fillna(0).copy()
                temp_fund.rename(columns={eps_col: '基本每股盈餘（元）', rev_col: '營業收入', yr_col: '年度', q_col: '季別'}, inplace=True)
                for essential in ['年度', '季別', '基本每股盈餘（元）', '營業收入']:
                    if essential not in temp_fund.columns: temp_fund[essential] = '-'
                res['fund_data'] = temp_fund.to_dict(orient='records')

        return jsonify(res)
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': f'後端 AI 系統發生錯誤: {str(e)}'})

# ================= 網頁前端 (HTML + JS) =================
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="zh-TW">
<head>
    <meta charset="UTF-8">
    <title>🚀 極速海選 x AI 深度決策系統</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdn.datatables.net/1.13.4/css/dataTables.bootstrap5.min.css" rel="stylesheet">
    <style>
        body { font-family: "Microsoft JhengHei", sans-serif; background-color: #f4f6f9; }
        .card-header { background-color: #2c3e50; color: white; font-weight: bold; }
        .text-up { color: #e74c3c; font-weight: bold; }
        .text-down { color: #27ae60; font-weight: bold; }
        .tag-box { font-weight: bold; padding: 4px 8px; border-radius: 4px; display: inline-block; font-size: 0.85rem; margin: 2px; }
        .tag-sediment { background-color: #d5f5e3; color: #27ae60; border: 1px solid #2ecc71; }
        .tag-squeeze { background-color: #fadbd8; color: #c0392b; border: 1px solid #e74c3c; }
        .tag-strict { background-color: #fcf3cf; color: #b7950b; border: 1px solid #f1c40f; }
        
        .stock-trigger { color: #0d6efd; cursor: pointer; text-decoration: underline; font-weight: bold; font-size: 1.1em;}
        .stock-trigger:hover { color: #d32f2f; }
        
        .metric-card { background: #fff3e0; border-left: 5px solid #ff9800; padding: 20px; margin: 15px 0; border-radius: 4px; text-align: center; box-shadow: 0 2px 5px rgba(0,0,0,0.05); }
        .ai-table th { background-color: #f8f9fa; color: #333; text-align:center;}
        .ai-table td { text-align:center; vertical-align: middle;}
        .alert-badge { display: inline-block; padding: 4px 8px; border-radius: 4px; font-size: 12px; font-weight: bold; margin-top: 4px; }
        .alert-danger { background-color: #ffebee; color: #c62828; border: 1px solid #ffcdd2; }
        .alert-warning { background-color: #fff8e1; color: #f57f17; border: 1px solid #ffecb3; }
        .alert-success { background-color: #e8f5e9; color: #2e7d32; border: 1px solid #c8e6c9; }
        .spinner-border { width: 3rem; height: 3rem; margin: 20px; color: #0d6efd; }
    </style>
</head>
<body>
    <div class="container-fluid mt-3">
        <h3 class="text-center mb-3">🚀 極速海選雷達 x AI 決策大腦</h3>
        
        <div class="card mb-3 shadow-sm">
            <div class="card-header">🛠️ 第一關：SQL 極速參數設定 (表格掃描)</div>
            <div class="card-body">
                <form method="POST">
                    <div class="row g-3">
                        <div class="col-md-2">
                            <label class="form-label fw-bold">1. 基準日期</label>
                            <input type="text" class="form-control form-control-sm" name="target_date" id="target_date" value="{{ target_date }}">
                        </div>
                        <div class="col-md-2">
                            <label class="form-label fw-bold">2. 最低漲幅 (%)</label>
                            <input type="number" step="0.1" class="form-control form-control-sm" name="min_price_change" value="{{ criteria.min_price_change }}">
                        </div>
                        <div class="col-md-2">
                            <label class="form-label fw-bold">3. 法人買超 (張)</label>
                            <input type="number" class="form-control form-control-sm" name="min_inst_buy" value="{{ criteria.min_inst_buy }}">
                        </div>
                        <div class="col-md-3">
                            <label class="form-label fw-bold">4. 篩選邏輯模式</label>
                            <select class="form-select form-select-sm" name="strategy_mode">
                                <option value="all" {% if criteria.strategy_mode == 'all' %}selected{% endif %}>全部顯示 (只要符合其一)</option>
                                <option value="strict_and" {% if criteria.strategy_mode == 'strict_and' %}selected{% endif %}>🎯 嚴格篩選 (資減 + 券增)</option>
                            </select>
                        </div>
                        <div class="col-md-3 d-flex align-items-end">
                            <button type="submit" class="btn btn-primary btn-sm w-100">🔍 啟動第一關掃描</button>
                        </div>
                        <div class="col-md-3">
                            <label class="form-label text-success fw-bold">5. 融資減幅小於 (%)</label>
                            <input type="number" step="0.1" class="form-control form-control-sm" name="margin_th" value="{{ criteria.margin_th }}">
                        </div>
                        <div class="col-md-3">
                            <label class="form-label text-danger fw-bold">6. 融券增幅大於 (%)</label>
                            <input type="number" step="0.1" class="form-control form-control-sm" name="short_th" value="{{ criteria.short_th }}">
                        </div>
                    </div>
                </form>
            </div>
        </div>

        <!-- 🌟 新增的防呆顯示邏輯 -->
        {% if searched %}
            {% if results %}
                <ul class="nav nav-tabs" id="myTab">
                    {% for days, df in results.items() %}
                    <li class="nav-item">
                        <button class="nav-link {% if loop.first %}active{% endif %}" data-bs-toggle="tab" data-bs-target="#content-{{ days }}" type="button">近 {{ days }} 日</button>
                    </li>
                    {% endfor %}
                </ul>
                <div class="tab-content bg-white border border-top-0 p-3 shadow-sm mb-5">
                    {% for days, df in results.items() %}
                    <div class="tab-pane fade {% if loop.first %}show active{% endif %}" id="content-{{ days }}">
                        <table class="table table-striped table-bordered w-100 datatable">
                            <thead>
                                <tr>
                                    <th>代號 (點擊啟動 AI)</th><th>名稱</th><th>收盤價</th><th>漲跌%</th><th>符合條件</th>
                                    <th>資增減%</th><th>券增減%</th>
                                    <th>法人合計(股)</th>
                                    <th>備註</th>
                                </tr>
                            </thead>
                            <tbody>
                                {% for row in df %}
                                <tr>
                                    <td class="text-center"><span class="stock-trigger" data-id="{{ row['證券代號'] }}" data-name="{{ row['證券名稱'] }}">🎯 {{ row['證券代號'] }}</span></td>
                                    <td>{{ row['證券名稱'] }}</td>
                                    <td>{{ row['收盤價'] }}</td>
                                    <td class="{{ 'text-up' if row['股價漲跌幅%'] > 0 else 'text-down' }}">{{ "%.2f"|format(row['股價漲跌幅%']) }}%</td>
                                    <td>{{ row['籌碼預測_HTML'] | safe }}</td>
                                    <td class="{{ 'text-up' if row['融資增減幅%'] > 0 else 'text-down' }}">{{ "%.2f"|format(row['融資增減幅%']) }}%</td>
                                    <td class="{{ 'text-up' if row['融券增減幅%'] > 0 else 'text-down' }}">{{ "%.2f"|format(row['融券增減幅%']) }}%</td>
                                    <td class="{{ 'text-up' if row['法人區間買賣'] > 0 else 'text-down' }} fw-bold">{{ row['法人區間買賣_顯示'] }}</td>
                                    <td class="text-secondary">{{ row['備註分析'] | safe }}</td>
                                </tr>
                                {% endfor %}
                            </tbody>
                        </table>
                    </div>
                    {% endfor %}
                </div>
            {% else %}
                <!-- 如果搜尋了卻沒有結果，顯示這個警告框 -->
                <div class="alert alert-warning text-center mt-4 shadow-sm" style="border-left: 5px solid #f39c12;">
                    <h4 class="fw-bold">📭 找不到符合條件的標的，或資料尚未同步！</h4>
                    <p class="mb-0">這段期間內沒有股票滿足設定條件。<br>
                    💡 小提醒：目前【三大法人】與【融資融券】最新資料停留在 <strong>20251023</strong>，請使用此日期測試，或更新資料庫！</p>
                </div>
            {% endif %}
        {% endif %}
    </div>

    <!-- 超大型 Modal：AI 深度決策大腦 -->
    <div class="modal fade" id="aiModal" tabindex="-1" aria-hidden="true">
        <div class="modal-dialog modal-xl modal-dialog-centered modal-dialog-scrollable">
            <div class="modal-content">
                <div class="modal-header bg-dark text-white">
                    <h4 class="modal-title fw-bold" id="aiModalTitle">🧠 AI 深度決策解析</h4>
                    <button type="button" class="btn-close btn-close-white" data-bs-dismiss="modal"></button>
                </div>
                <div class="modal-body p-4" id="aiModalBody">
                    <div id="ai-loading" class="text-center py-5">
                        <div class="spinner-border" role="status"></div>
                        <h4 class="mt-3 text-secondary">正在喚醒 100 棵決策樹，計算專屬動態門檻...</h4>
                    </div>
                    <div id="ai-results" style="display:none;"></div>
                </div>
            </div>
        </div>
    </div>

    <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
    <script src="https://cdn.datatables.net/1.13.4/js/jquery.dataTables.min.js"></script>
    <script src="https://cdn.datatables.net/1.13.4/js/dataTables.bootstrap5.min.js"></script>
    <script>
        $(document).ready(function() {
            $('.datatable').DataTable({ "order": [[ 7, "desc" ]], "language": { "url": "//cdn.datatables.net/plug-ins/1.13.4/i18n/zh-HANT.json" } });
            
            $(document).on('click', '.stock-trigger', async function() {
                const stockId = $(this).data('id');
                const stockName = $(this).data('name');
                const targetDate = $('#target_date').val();
                
                $('#aiModalTitle').text(`🧠 【${stockId} ${stockName}】 AI 決策大腦解析中`);
                $('#ai-loading').show();
                $('#ai-results').hide().empty();
                
                new bootstrap.Modal(document.getElementById('aiModal')).show();
                
                try {
                    const response = await fetch('/api/predict', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ stock_id: stockId, target_date: targetDate }) 
                    });
                    const d = await response.json();
                    
                    if(d.error) {
                        $('#ai-loading').hide();
                        $('#ai-results').html(`<h3 class="text-danger text-center mt-4">${d.error}</h3>`).show();
                        return;
                    }
                    
                    let h = `
                    <div class="metric-card">
                        <h5>🎯 明日 AI 勝率預測</h5>
                        <div class="d-flex justify-content-center gap-5 my-3">
                            <div style="font-size: 32px; font-weight: bold; color: #d32f2f;">📈 看漲: ${d.prob_up}%</div>
                            <div style="font-size: 32px; font-weight: bold; color: #388e3c;">📉 看跌/平: ${d.prob_down}%</div>
                        </div>
                        <p class="text-muted mb-0">( 依據最新特徵演算 | 模型歷史預測準確率: ${d.accuracy}% | 訓練池: ${d.train_count} 天 )</p>
                    </div>`;

                    h += '<h5 class="mt-4 border-start border-4 border-primary ps-2 fw-bold">📊 近四日【進階籌碼與動能指標】</h5>';
                    h += '<table class="table table-bordered table-hover ai-table mt-3"><thead><tr><th>日期</th><th>收盤價</th><th>漲跌%</th><th>法人買賣超(股)</th><th>融資增減%</th><th>資券比%</th><th>估算維持率</th><th>系統判定狀態</th></tr></thead><tbody>';
                    
                    d.recent_data.forEach(r => { 
                        let alertHtml = (r.籌碼警示 || '').split(' ').map(tag => {
                            if(!tag) return '';
                            if (tag.includes('接刀') || tag.includes('斷頭')) return `<span class="alert-badge alert-danger">${tag}</span>`;
                            if (tag.includes('軋空')) return `<span class="alert-badge alert-warning">${tag}</span>`;
                            if (tag.includes('沉澱')) return `<span class="alert-badge alert-success">${tag}</span>`;
                            return `<span class="alert-badge">${tag}</span>`;
                        }).join('<br>');

                        h += `<tr>
                            <td class="text-muted">${r.日期}</td>
                            <td class="fw-bold fs-6">${r.收盤價}</td>
                            <td class="${r['股價日增減(%)']>0 ? 'text-up' : 'text-down'}">${r['股價日增減(%)'].toFixed(2)}%</td>
                            <td class="${r.三大法人買賣超股數>0 ? 'text-up' : 'text-down'} fw-bold">${Number(r.三大法人買賣超股數).toLocaleString()}</td>
                            <td>${r['融資日增減(%)'].toFixed(2)}%</td>
                            <td class="text-primary fw-bold">${r['資券比(%)'].toFixed(2)}%</td>
                            <td>${r['估算維持率(%)'].toFixed(1)}%</td>
                            <td>${alertHtml || '-'}</td>
                        </tr>`; 
                    });
                    h += '</tbody></table>';

                    if(d.feature_analysis && d.feature_analysis.length > 0) {
                        h += '<h5 class="mt-4 border-start border-4 border-danger ps-2 fw-bold">🧠 AI 決策核心依據 (特徵權重解析)</h5>';
                        h += '<table class="table table-bordered ai-table mt-3"><thead><tr><th>特徵指標</th><th>最新實際數據 (預測基準)</th><th>AI 判定權重占比</th></tr></thead><tbody>';
                        d.feature_analysis.forEach(f => {
                            let valStr = f.value;
                            if (f.feature.includes('股數') || f.feature.includes('差')) valStr = Number(f.value).toLocaleString();
                            else if (f.feature.includes('%') || f.feature.includes('率') || f.feature.includes('比')) valStr += '%';
                            
                            h += `<tr>
                                <td class="fw-bold text-start ps-3 text-secondary">${f.feature}</td>
                                <td class="text-primary fw-bold">${valStr}</td>
                                <td>
                                    <div class="d-flex align-items-center justify-content-center gap-2">
                                        <div class="progress" style="width: 150px; height: 10px;">
                                            <div class="progress-bar bg-danger" style="width: ${f.weight}%"></div>
                                        </div>
                                        <span class="text-muted" style="width:50px; text-align:right;">${f.weight}%</span>
                                    </div>
                                </td>
                            </tr>`;
                        });
                        h += '</tbody></table>';
                    }

                    if(d.fund_data && d.fund_data.length > 0) {
                        h += '<h5 class="mt-4 border-start border-4 border-success ps-2 fw-bold">🏢 基本面追蹤</h5>';
                        h += '<table class="table table-bordered ai-table mt-3"><thead><tr><th>年度</th><th>季別</th><th>EPS (元)</th><th>營業收入</th></tr></thead><tbody>';
                        d.fund_data.forEach(f => { 
                            h += `<tr><td>${f.年度}</td><td>${f.季別}</td><td class="text-danger fw-bold fs-6">${f['基本每股盈餘（元）']}</td><td>${Number(f.營業收入).toLocaleString()}</td></tr>`; 
                        });
                        h += '</tbody></table>';
                    }

                    $('#ai-loading').hide();
                    $('#ai-results').html(h).fadeIn();

                } catch(e) {
                    $('#ai-loading').hide();
                    $('#ai-results').html(`<h3 class="text-danger text-center mt-4">前端解析失敗: ${e.message}</h3>`).show();
                }
            });
        });
    </script>
</body>
</html>
"""

if __name__ == "__main__":
    app.run(debug=True, port=5000)