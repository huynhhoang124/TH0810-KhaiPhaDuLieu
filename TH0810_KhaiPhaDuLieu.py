#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from IPython.display import display, Markdown
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, TimeSeriesSplit
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay,
                             silhouette_score, adjusted_rand_score, normalized_mutual_info_score)

pd.set_option('display.max_columns', 60); pd.set_option('display.width', 200)
pd.set_option('display.float_format', lambda v: f'{v:,.3f}')
plt.rcParams.update({'figure.dpi': 100, 'axes.grid': True, 'grid.alpha': .3, 'axes.axisbelow': True})
SEED = 42
COLORS = ['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B3', '#937860']
short = lambda c: c.replace('Curricular units ', 'CU ')

def save(name):
    plt.tight_layout(); plt.savefig(f'figures/{name}.png', dpi=150, bbox_inches='tight'); plt.show()

def stats_table(df):
    # describe + skewness + kurtosis + số giá trị thiếu
    t = df.describe().T
    t['skew'] = df.skew(); t['kurtosis'] = df.kurtosis(); t['missing'] = df.isna().sum()
    return t.rename(columns={'count': 'n', '50%': 'median'}, index=short)

MODELS = {
    'KNN': (KNeighborsClassifier(),
            {'clf__n_neighbors': list(range(1, 32, 2)), 'clf__weights': ['uniform', 'distance']}),
    'Naive Bayes': (GaussianNB(), {'clf__var_smoothing': np.logspace(-12, 0, 13)}),
    'Decision Tree': (DecisionTreeClassifier(random_state=SEED),
                      {'clf__max_depth': [2, 3, 4, 5, 6, 8, 10, 15, None],
                       'clf__min_samples_leaf': [1, 5, 10, 20, 50]}),
}

def classify(pre, X_tr, y_tr, X_te, y_te, cv, labels, tag):
    # Huấn luyện 3 mô hình, in bảng kết quả, vẽ đường k của KNN, ma trận nhầm lẫn và biểu đồ so sánh
    fits, rows, preds = {}, [], {}
    for name, (clf, grid) in MODELS.items():
        gs = GridSearchCV(Pipeline([('pre', pre), ('clf', clf)]), grid, cv=cv,
                          scoring='f1_macro', n_jobs=-1).fit(X_tr, y_tr)
        p = preds[name] = gs.predict(X_te); fits[name] = gs
        best = {k.replace('clf__', ''): (float(f'{v:.0e}') if isinstance(v, float) else v)
                for k, v in gs.best_params_.items()}
        rows.append({'Mô hình': name, 'Tham số tốt nhất': str(best), 'CV macro-F1': gs.best_score_,
                     'Accuracy': accuracy_score(y_te, p), 'macro-F1': f1_score(y_te, p, average='macro'),
                     **{f'F1 {l}': v for l, v in zip(labels, f1_score(y_te, p, labels=labels, average=None))}})
    res = pd.DataFrame(rows).set_index('Mô hình')

    r = pd.DataFrame(fits['KNN'].cv_results_)
    fig, ax = plt.subplots(figsize=(7, 3))
    for w, c in zip(['uniform', 'distance'], COLORS):
        s = r[r.param_clf__weights == w]
        ax.plot(s.param_clf__n_neighbors, s.mean_test_score, 'o-', ms=4, color=c, label=f'weights={w}')
    ax.set_xlabel('k'); ax.set_ylabel('CV macro-F1'); ax.legend(); ax.set_title('KNN – chọn k bằng 5-fold CV')
    save(f'{tag}_knn_k')

    display(Markdown('**Kết quả trên tập test** (tham số chọn bằng GridSearchCV trên tập train)')); display(res)
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.9))
    for ax, (name, p) in zip(axes, preds.items()):
        ConfusionMatrixDisplay(confusion_matrix(y_te, p, labels=labels), display_labels=labels).plot(
            ax=ax, cmap='Blues', colorbar=False, values_format='d')
        ax.set_title(name); ax.grid(False)
    fig.suptitle('Ma trận nhầm lẫn trên tập test', y=1.02); save(f'{tag}_cm')
    return fits, res

def importance(gs, names, tag, top=10):
    imp = pd.Series(gs.best_estimator_['clf'].feature_importances_, index=names).sort_values().tail(top)
    fig, ax = plt.subplots(figsize=(7, .32 * len(imp) + .8)); ax.barh(imp.index, imp.values, color=COLORS[2])
    ax.set_title('Decision Tree – độ quan trọng đặc trưng (Gini)'); save(f'{tag}_tree_importance')

def kmeans_scan(X, tag, ks=range(2, 11)):
    rows = []
    for k in ks:
        km = KMeans(k, init='k-means++', n_init=10, random_state=SEED).fit(X)
        rows.append({'k': k, 'Inertia': km.inertia_, 'Silhouette': silhouette_score(X, km.labels_)})
    r = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3))
    axes[0].plot(r.k, r.Inertia, 'o-', color=COLORS[0]); axes[0].set_title('Elbow (Inertia)'); axes[0].set_xlabel('k')
    axes[1].plot(r.k, r.Silhouette, 'o-', color=COLORS[1]); axes[1].set_title('Silhouette'); axes[1].set_xlabel('k')
    save(f'{tag}_kmeans_elbow'); display(r.set_index('k').T)


# In[2]:


df = pd.read_csv('data/697/data.csv', encoding='utf-8-sig'); df.columns = df.columns.str.strip()
print(f'Kích thước: {df.shape[0]:,} dòng × {df.shape[1]} cột | Giá trị thiếu: {df.isna().sum().sum()} | '
      f'Dòng trùng: {df.duplicated().sum()}')
CAT_A = ['Marital Status', 'Application mode', 'Course', 'Previous qualification', 'Nacionality',
         "Mother's qualification", "Father's qualification", "Mother's occupation", "Father's occupation"]
BIN_A = ['Daytime/evening attendance', 'Displaced', 'Educational special needs', 'Debtor',
         'Tuition fees up to date', 'Gender', 'Scholarship holder', 'International']
NUM_A = [c for c in df.columns if c not in CAT_A + BIN_A + ['Target']]
SHOW_A = ['Admission grade', 'Previous qualification (grade)', 'Age at enrollment',
          'Curricular units 1st sem (credited)', 'Curricular units 1st sem (approved)',
          'Curricular units 1st sem (grade)', 'Curricular units 2nd sem (approved)',
          'Curricular units 2nd sem (grade)', 'Unemployment rate']
display(Markdown('**Thống kê mô tả các biến số chính**'))
display(stats_table(df[SHOW_A]))

order = ['Dropout', 'Enrolled', 'Graduate']; cmap = {'Dropout': COLORS[3], 'Enrolled': COLORS[1], 'Graduate': COLORS[2]}
fig, axes = plt.subplots(1, 4, figsize=(15, 3.6))
tc = df.Target.value_counts()[order]
b = axes[0].bar(tc.index, tc.values, color=[cmap[t] for t in order])
axes[0].bar_label(b, labels=[f'{v:,}\n({v / len(df):.0%})' for v in tc.values], fontsize=8)
axes[0].set_title('Phân bố nhãn Target'); axes[0].set_ylim(0, tc.max() * 1.25)
for ax, c in zip(axes[1:], ['Admission grade', 'Age at enrollment', 'Curricular units 2nd sem (approved)']):
    ax.boxplot([df.loc[df.Target == t, c] for t in order], labels=order, patch_artist=True,
               boxprops=dict(facecolor='#c6d4ea'))
    ax.set_title(short(c), fontsize=10)
save('A_eda')

display(Markdown('**Trung bình theo lớp và tương quan với Target** (Dropout=0, Enrolled=1, Graduate=2)'))
tmp = df[NUM_A + BIN_A].assign(Target_code=df.Target.map({'Dropout': 0, 'Enrolled': 1, 'Graduate': 2}))
r = tmp.corr()['Target_code'].drop('Target_code'); top = r.abs().sort_values(ascending=False).index[:8]
t = df.groupby('Target')[list(top)].mean().T; t['r với Target'] = r[top]
display(t.rename(index=short))


# In[3]:


X, y = df.drop(columns='Target'), df.Target
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=.2, stratify=y, random_state=SEED)
pre_A = ColumnTransformer([('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), CAT_A),
                           ('num', StandardScaler(), NUM_A + BIN_A)])
print(f'Train: {len(X_tr):,} mẫu | Test: {len(X_te):,} mẫu | '
      f'Số đặc trưng sau One-Hot: {pre_A.fit(X_tr).transform(X_tr[:1]).shape[1]}')


# In[4]:


fits_A, res_A = classify(pre_A, X_tr, y_tr, X_te, y_te, StratifiedKFold(5, shuffle=True, random_state=SEED),
                         ['Dropout', 'Enrolled', 'Graduate'], 'A')
importance(fits_A['Decision Tree'], [short(n.split('__')[1]) for n in
           fits_A['Decision Tree'].best_estimator_['pre'].get_feature_names_out()], 'A')


# In[5]:


KM_A = ['Previous qualification (grade)', 'Admission grade', 'Age at enrollment'] + \
       [c for c in df.columns if c.startswith('Curricular')]
Z_A = StandardScaler().fit_transform(df[KM_A]); kmeans_scan(Z_A, 'A')

km = KMeans(3, init='k-means++', n_init=10, random_state=SEED).fit(Z_A); cl = pd.Series(km.labels_, name='Cụm')
print(f'K = 3: Silhouette = {silhouette_score(Z_A, cl):.3f} | ARI = {adjusted_rand_score(df.Target, cl):.3f} | '
      f'NMI = {normalized_mutual_info_score(df.Target, cl):.3f}')
pc = PCA(2, random_state=SEED).fit(Z_A); P = pc.transform(Z_A); C = pc.transform(km.cluster_centers_)
fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
for k in range(3):
    axes[0].scatter(*P[cl == k].T, s=5, alpha=.5, color=COLORS[k], label=f'Cụm {k}')
axes[0].scatter(*C.T, marker='X', s=150, c='black'); axes[0].set_title('K-Means k=3 (X = tâm cụm)')
for t in ['Dropout', 'Enrolled', 'Graduate']:
    axes[1].scatter(*P[df.Target == t].T, s=5, alpha=.5, color=cmap[t], label=t)
axes[1].set_title('Nhãn thật Target')
for ax in axes:
    ax.set_xlabel('PC1'); ax.set_ylabel('PC2'); ax.legend(markerscale=3, fontsize=8)
save('A_kmeans_pca')

prof = df[['Age at enrollment', 'Curricular units 1st sem (credited)', 'Curricular units 2nd sem (enrolled)',
           'Curricular units 2nd sem (approved)', 'Curricular units 2nd sem (grade)']].groupby(cl).mean()
ct = pd.crosstab(cl, df.Target, normalize='index') * 100
display(Markdown('**Đặc điểm từng cụm (trung bình) và tỉ lệ % Target trong cụm**'))
display(pd.concat([prof.rename(columns=short), ct.add_suffix(' (%)')], axis=1).assign(**{'Số SV': cl.value_counts().sort_index()}))


# In[6]:


pw = pd.read_csv('data/235/household_power_consumption.txt', sep=';', na_values='?', low_memory=False)
pw.index = pd.to_datetime(pw.pop('Date') + ' ' + pw.pop('Time'), format='%d/%m/%Y %H:%M:%S')
pw = pw.astype('float64'); miss = pw.isna().any(axis=1)
print(f'Kích thước: {pw.shape[0]:,} dòng × {pw.shape[1]} biến | {pw.index.min():%d/%m/%Y} → {pw.index.max():%d/%m/%Y}')
print(f'Dòng thiếu giá trị: {miss.sum():,} ({miss.mean():.2%}), nằm trong {pw.index[miss].normalize().nunique()} ngày')
pw['Sub_metering_rest'] = pw.Global_active_power * 1000 / 60 - pw[['Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']].sum(axis=1, min_count=3)
display(Markdown('**Thống kê mô tả (dữ liệu theo phút)**')); display(stats_table(pw))

gap = pw.Global_active_power
energy = pw[['Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3', 'Sub_metering_rest']].sum()
fig, axes = plt.subplots(1, 3, figsize=(15, 3.6))
axes[0].hist(gap.dropna(), bins=100, color=COLORS[0]); axes[0].set_title('Histogram GAP (kW)')
axes[1].pie(energy, labels=['Bếp', 'Giặt', 'Nóng lạnh + ĐH', 'Khác'], autopct='%1.0f%%', colors=COLORS[:4], startangle=90)
axes[1].set_title('Cơ cấu điện năng')
c = pw.corr(); im = axes[2].imshow(c, cmap='RdBu_r', vmin=-1, vmax=1)
lab = ['GAP', 'Reactive', 'Voltage', 'Intensity', 'S1', 'S2', 'S3', 'Rest']
axes[2].set_xticks(range(8)); axes[2].set_xticklabels(lab, rotation=90, fontsize=8)
axes[2].set_yticks(range(8)); axes[2].set_yticklabels(lab, fontsize=8); axes[2].grid(False)
for i in range(8):
    for j in range(8):
        axes[2].text(j, i, f'{c.iat[i, j]:.2f}', ha='center', va='center', fontsize=6, color='white' if abs(c.iat[i, j]) > .6 else 'black')
axes[2].set_title('Tương quan Pearson'); save('B_eda')

days = ['T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'CN']
fig, axes = plt.subplots(1, 3, figsize=(15, 3.4))
for wk, lab, col in [(False, 'Ngày thường', COLORS[0]), (True, 'Cuối tuần', COLORS[1])]:
    s = gap[(gap.index.dayofweek >= 5) == wk]; axes[0].plot(s.groupby(s.index.hour).mean(), 'o-', ms=3, color=col, label=lab)
axes[0].set_title('GAP theo giờ trong ngày'); axes[0].legend()
axes[1].bar(days, gap.groupby(gap.index.dayofweek).mean(), color=COLORS[2]); axes[1].set_title('GAP theo thứ')
axes[2].bar(range(1, 13), gap.groupby(gap.index.month).mean(), color=COLORS[4]); axes[2].set_title('GAP theo tháng')
axes[2].set_xticks(range(1, 13)); save('B_profiles')


# In[7]:


H = pw.drop(columns='Sub_metering_rest').resample('h').mean()
H['lag1'] = H.Global_active_power.shift(1); H = H.dropna()
H['hour'], H['dayofweek'], H['month'] = H.index.hour, H.index.dayofweek, H.index.month
H['is_weekend'] = (H.dayofweek >= 5).astype(int)
cut = int(len(H) * .8); q1, q2 = H.Global_active_power.iloc[:cut].quantile([1 / 3, 2 / 3])
LAB_B = ['Thấp', 'Trung bình', 'Cao']
H['Mức tải'] = pd.cut(H.Global_active_power, [-np.inf, q1, q2, np.inf], labels=LAB_B).astype(str)
FEAT_B = ['hour', 'dayofweek', 'month', 'is_weekend', 'Voltage', 'Global_reactive_power', 'lag1']
Xb_tr, Xb_te, yb_tr, yb_te = H[FEAT_B].iloc[:cut], H[FEAT_B].iloc[cut:], H['Mức tải'].iloc[:cut], H['Mức tải'].iloc[cut:]
print(f'Số giờ: {len(H):,} | Ngưỡng: Thấp ≤ {q1:.3f} kW < Trung bình ≤ {q2:.3f} kW < Cao')
print(f'Train: {Xb_tr.index.min():%d/%m/%Y} → {Xb_tr.index.max():%d/%m/%Y} ({len(Xb_tr):,} giờ) | '
      f'Test: {Xb_te.index.min():%d/%m/%Y} → {Xb_te.index.max():%d/%m/%Y} ({len(Xb_te):,} giờ)')
display(pd.DataFrame({'Train (%)': yb_tr.value_counts(normalize=True) * 100,
                      'Test (%)': yb_te.value_counts(normalize=True) * 100}).loc[LAB_B].T)


# In[8]:


fits_B, res_B = classify(StandardScaler(), Xb_tr, yb_tr, Xb_te, yb_te, TimeSeriesSplit(5), LAB_B, 'B')
importance(fits_B['Decision Tree'], FEAT_B, 'B')


# In[9]:


D = H.Global_active_power.to_frame('gap').assign(date=H.index.normalize(), hour=H.index.hour) \
     .pivot_table(index='date', columns='hour', values='gap').dropna()
print(f'Số ngày đủ 24 giờ: {len(D):,}'); kmeans_scan(D.values, 'B')

K_B = 4
km = KMeans(K_B, init='k-means++', n_init=10, random_state=SEED).fit(D.values)
order = np.argsort(km.cluster_centers_.mean(axis=1)); remap = {o: n for n, o in enumerate(order)}
cl = pd.Series(km.labels_, index=D.index).map(remap); centers = km.cluster_centers_[order]
print(f'K = {K_B}: Silhouette = {silhouette_score(D.values, cl):.3f}')

season = cl.index.month.map({12: 'Đông', 1: 'Đông', 2: 'Đông', 3: 'Xuân', 4: 'Xuân', 5: 'Xuân',
                             6: 'Hè', 7: 'Hè', 8: 'Hè', 9: 'Thu', 10: 'Thu', 11: 'Thu'})
dow = np.array(days)[cl.index.dayofweek]
fig, axes = plt.subplots(1, 3, figsize=(15, 3.8))
for k in range(K_B):
    axes[0].plot(centers[k], color=COLORS[k], lw=2.5, label=f'Cụm {k} ({(cl == k).sum()} ngày)')
axes[0].set_title('Tâm cụm = biểu đồ phụ tải 24h'); axes[0].set_xlabel('Giờ'); axes[0].set_ylabel('kW'); axes[0].legend(fontsize=8)
for ax, by, idx, title in [(axes[1], dow, days, 'thứ'), (axes[2], season, ['Xuân', 'Hè', 'Thu', 'Đông'], 'mùa')]:
    (pd.crosstab(np.asarray(by), cl.values, normalize='index') * 100).loc[idx].plot.bar(
        stacked=True, ax=ax, color=COLORS[:K_B], rot=0, width=.75, legend=False)
    ax.set_title(f'Tỉ lệ (%) các cụm theo {title}'); ax.set_xlabel('')
axes[2].legend(title='Cụm', fontsize=8, bbox_to_anchor=(1, 1)); save('B_kmeans')

display(Markdown('**Đặc điểm từng cụm**'))
display(pd.DataFrame({'Số ngày': cl.value_counts().sort_index(), 'GAP TB (kW)': centers.mean(axis=1),
                      'GAP đỉnh (kW)': centers.max(axis=1), 'Giờ đỉnh': centers.argmax(axis=1),
                      'kWh/ngày': centers.sum(axis=1),
                      '% cuối tuần': pd.Series(cl.index.dayofweek >= 5, index=cl.index).groupby(cl).mean() * 100}))

