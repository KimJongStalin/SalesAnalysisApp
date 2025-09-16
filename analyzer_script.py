# import pandas as pd
# import numpy as np
# from statsmodels.tsa.statespace.sarimax import SARIMAX
# import json
# from datetime import datetime
# from scipy.stats.mstats import winsorize
# from itertools import combinations
# import statsmodels.api as sm

# class SalesAnalyzer:
#     """
#     一个用于处理和分析销售数据的可复用、配置驱动的工具。
#     专注于生成一个单一、全面、高度交互的HTML仪表板。(V9.6 稳定产品标识版)
#     """

#     def __init__(self, config: dict):
#         self.config = config
#         self.df = None
#         print("✅ SalesAnalyzer 初始化成功。")

#     def load_and_preprocess_data(self) -> bool:
#         file_path = self.config.get('input_filepath')
#         cols = self.config['columns']
#         print(f"\n--- 正在从 '{file_path}' 加载数据 ---")
#         try:
#             self.df = pd.read_excel(file_path)
#             print(f"  - 初始加载了 {len(self.df)} 行数据")
#             print("加载的列: ", self.df.columns.tolist())
#         except Exception as e:
#             print(f"❌ 无法加载或解析 Excel 文件: {e}")
#             import traceback
#             traceback.print_exc()
#             return False

#         date_col = cols['date']
#         try:
#             self.df[date_col] = pd.to_datetime(self.df[date_col], format='%Y%m')
#             print("日期成功按 'YYYYMM' 格式解析。")
#         except (ValueError, TypeError):
#             print("按 'YYYYMM' 格式解析失败，正在尝试自动识别标准日期格式...")
#             self.df[date_col] = pd.to_datetime(self.df[date_col], errors='coerce')

#         self.df.dropna(subset=[date_col], inplace=True)
#         print(f"  - 清理无效日期后，剩下 {len(self.df)} 行数据")
#         if not self.df.empty:
#             unique_quarters = self.df[date_col].dt.to_period('Q').nunique()
#             print(f"  - 剩余数据覆盖了 {unique_quarters} 个不重复的季度")

#         sales_col = cols['sales']
#         self.df[sales_col] = pd.to_numeric(self.df[sales_col], errors='coerce').fillna(0)
#         self.df.sort_values(date_col, inplace=True)

#         if self.df.empty:
#             print("⚠️ 警告: 处理后没有剩下有效的数据。")
#             return False

#         first_listed_date_col = cols.get('first_listed_date')
#         if first_listed_date_col and first_listed_date_col in self.df.columns:
#             self.df[first_listed_date_col] = pd.to_datetime(self.df[first_listed_date_col], errors='coerce')

#         print("✅ 数据加载与预处理成功。")
#         return True

#     def prepare_html_report_data(self) -> dict:
#         """为交互式仪表板准备一个结构化的、JSON兼容的数据集合。"""
#         print("\n--- 正在准备所有分析数据 ---")
#         if self.df is None or self.df.empty:
#             return {}

#         cols = self.config['columns']
#         date_col, sales_col, type_col, asin_col = cols['date'], cols['sales'], cols['type'], cols['asin']
#         product_types = ["Overall"] + sorted(self.df[type_col].unique().tolist())

#         dynamic_time_events = []
#         year_agnostic_events = self.config.get("time_events", {})
#         if not self.df.empty and year_agnostic_events:
#             first_data_date = self.df[date_col].min()
#             last_year_in_data = self.df[date_col].max().year
#             for year in range(first_data_date.year, last_year_in_data + 2):
#                 for event_name, mm_dd in year_agnostic_events.items():
#                     full_date_str = f"{year}-{mm_dd}"
#                     try:
#                         event_date = pd.to_datetime(full_date_str)
#                         if event_date >= first_data_date:
#                             dynamic_time_events.append({"label": event_name, "date": full_date_str})
#                     except ValueError:
#                         continue

#         # --- 模块2：销售预测 (使用 SARIMAX) ---
#         print("--- 正在计算销售预测 ---")
#         forecast_data = {}
#         for p_type in product_types:
#             df_filtered = self.df if p_type == "Overall" else self.df[self.df[type_col] == p_type]
#             monthly_sales = df_filtered.set_index(date_col)[sales_col].resample('M').sum()

#             if len(monthly_sales) < 24:
#                 continue

#             try:
#                 # SARIMA 模型代码
#                 model = SARIMAX(monthly_sales,
#                                 order=(1, 1, 1),
#                                 seasonal_order=(1, 1, 1, 12),
#                                 enforce_invertibility=False,
#                                 enforce_stationarity=False)

#                 model_fit = model.fit(disp=False)

#                 forecast_result = model_fit.get_forecast(steps=12)
#                 forecast_ci = forecast_result.conf_int()

#                 forecast_points = []
#                 for i in range(len(forecast_result.predicted_mean)):
#                     date = forecast_result.predicted_mean.index[i].strftime('%Y-%m-%d')
#                     value = forecast_result.predicted_mean.iloc[i]
#                     lower = forecast_ci.iloc[i, 0]
#                     upper = forecast_ci.iloc[i, 1]

#                     forecast_points.append({
#                         "x": date,
#                         "y": round(value, 0),
#                         "y_lower": round(lower, 0),
#                         "y_upper": round(upper, 0)
#                     })

#                 historical_points = [{'x': date.strftime('%Y-%m-%d'), 'y': value} for date, value in monthly_sales.items()]

#                 moving_average = monthly_sales.rolling(window=3).mean().dropna()
#                 moving_average_points = [{'x': date.strftime('%Y-%m-%d'), 'y': round(value, 0)} for date, value in moving_average.items()]

#                 all_data_series = pd.concat([monthly_sales, forecast_result.predicted_mean])
#                 x = np.arange(len(all_data_series))
#                 y = all_data_series.values
#                 z = np.polyfit(x, y, 1)
#                 p = np.poly1d(z)
#                 trend_line_points = [{'x': date.strftime('%Y-%m-%d'), 'y': round(p(i), 0)} for i, date in enumerate(all_data_series.index)]

#                 forecast_data[p_type] = {
#                     "historical": historical_points,
#                     "forecast": forecast_points,
#                     "moving_average": moving_average_points,
#                     "trend_line": trend_line_points
#                 }
#             except Exception as e:
#                 print(f"❌ 为 '{p_type}' 类型生成 SARIMAX 预测时发生错误: {e}")
#                 continue

#         print("--- 正在计算季度同比增长 ---")
#         sales_wide_q = self.df.groupby([pd.Grouper(key=date_col, freq='Q'), type_col])[sales_col].sum().unstack(type_col).fillna(0)
#         yoy_wide_q = sales_wide_q.pct_change(periods=4) * 100
#         quarterly_yoy_data = {"labels": sales_wide_q.index.to_period('Q').strftime('%YQ%q').tolist(), "sales_datasets": [{"label": str(col), "data": sales_wide_q[col].round(0).tolist()} for col in sales_wide_q.columns], "yoy_datasets": [{"label": str(col) + " YoY", "data": yoy_wide_q[col].where(pd.notna(yoy_wide_q[col]), None).round(1).tolist()} for col in yoy_wide_q.columns]}

#         print("--- 正在计算市场份额 ---")
#         share_dimensions = ['type', 'brand', 'packsize', 'pricerange', 'tiptype']
#         share_data = {}
#         for dim in share_dimensions:
#             dim_col_name = cols.get(dim)
#             if not dim_col_name or dim_col_name not in self.df.columns: continue
#             share_data[dim] = {}
#             for p_type in product_types:
#                 df_filtered = self.df if p_type == "Overall" else self.df[self.df[type_col] == p_type]
#                 if df_filtered.empty: continue
#                 freq = 'M' if dim == 'type' else 'Q'
#                 time_format = '%Y-%m' if dim == 'type' else '%YQ%q'
#                 top_n = 20
#                 if dim == 'brand' and df_filtered[dim_col_name].nunique() > top_n:
#                     total_sales = df_filtered.groupby(dim_col_name)[sales_col].sum()
#                     top_brands = total_sales.nlargest(top_n).index.tolist()
#                     df_with_others = df_filtered.copy()
#                     df_with_others[dim_col_name] = df_with_others[dim_col_name].apply(lambda x: x if x in top_brands else '其他 (Others)')
#                     data = df_with_others.groupby([pd.Grouper(key=date_col, freq=freq), dim_col_name])[sales_col].sum().unstack(dim_col_name).fillna(0)
#                 else:
#                     data = df_filtered.groupby([pd.Grouper(key=date_col, freq=freq), dim_col_name])[sales_col].sum().unstack(dim_col_name).fillna(0)
#                 if data.empty: continue
#                 if '其他 (Others)' in data.columns:
#                     other_col = data.pop('其他 (Others)')
#                     data['其他 (Others)'] = other_col
#                 sorted_columns = data.sum().sort_values(ascending=False).index
#                 data_sorted = data[sorted_columns]
#                 data_sum = data_sorted.sum(axis=1)
#                 safe_data_sum = data_sum.where(data_sum != 0, 1)
#                 data_pct_sorted = data_sorted.div(safe_data_sum, axis=0) * 100
#                 datasets = [{"label": str(col), "data": data_pct_sorted[col].round(1).tolist(), "absoluteData": data_sorted[col].round(0).tolist()} for col in data_sorted.columns]
#                 share_data[dim][p_type] = { "labels": data_pct_sorted.index.to_period(freq).strftime(time_format).tolist(), "datasets": datasets }

#         print("--- 正在计算增长表格 ---")
#         table_dimensions = {'brand': ['brand'], 'packsize': ['packsize'], 'pricerange': ['pricerange'], 'tiptype': ['tiptype'], 'tiptype_packsize': ['tiptype', 'packsize']}
#         table_data = {}
#         for key, dim_names in table_dimensions.items():
#             dim_cols = [cols.get(d) for d in dim_names if cols.get(d) in self.df.columns]
#             if len(dim_cols) != len(dim_names): continue
#             table_data[key] = {}
#             for p_type in product_types:
#                 df_filtered = self.df if p_type == "Overall" else self.df[self.df[type_col] == p_type]
#                 if df_filtered.empty: continue
#                 sales = df_filtered.groupby(dim_cols + [pd.Grouper(key=date_col, freq='Q')])[sales_col].sum().unstack(date_col).fillna(0)
#                 if sales.empty: continue
#                 if len(sales.columns) > 0:
#                     last_quarter_col = sales.columns[-1]
#                     sales = sales.sort_values(by=last_quarter_col, ascending=False)
#                 if key == 'brand' and len(sales) > 20:
#                     top_sales = sales.head(20)
#                     other_sales = sales.iloc[20:].sum()
#                     other_row = pd.DataFrame(other_sales).T
#                     other_row.index = ['其他 (Others)']
#                     sales = pd.concat([top_sales, other_row])
#                 if '其他 (Others)' in sales.index:
#                     other_row = sales.loc[['其他 (Others)']]
#                     sales = sales.drop('其他 (Others)')
#                     sales = pd.concat([sales, other_row])
#                 display_map = self.config.get("header_mappings", {})
#                 display_names = [display_map.get(name.lower(), name.capitalize()) for name in dim_names]
#                 display_header = " & ".join(display_names)
#                 headers = [display_header] + [q.to_period('Q').strftime('%YQ%q') for q in sales.columns]
#                 rows = []
#                 for index, row_data in sales.iterrows():
#                     row_content = [{'type': 'label', 'value': str(index)}]
#                     for i, quarter in enumerate(sales.columns):
#                         sale_val_num = row_data.get(quarter, 0)
#                         yoy_val, yoy_status = None, 'neutral'
#                         if sale_val_num == 0:
#                             sale_val_str = "-"
#                         else:
#                             sale_val_str = f"{(sale_val_num / 10000.0):.2f}万"
#                             if i >= 4:
#                                 prior_sale_val = row_data.get(sales.columns[i-4], 0)
#                                 if prior_sale_val > 0:
#                                     yoy = (sale_val_num / prior_sale_val) - 1
#                                     yoy_val, yoy_status = f"{yoy:.1%}", 'positive' if yoy > 0 else 'negative'
#                                 else:
#                                     yoy_val, yoy_status = "New", 'positive'
#                         row_content.append({'type': 'data', 'value': sale_val_str, 'yoy': yoy_val, 'yoy_status': yoy_status})
#                     rows.append(row_content)
#                 table_data[key][p_type] = {"headers": headers, "rows": rows}

#         print("--- 正在计算帕累托数据 ---")
#         pareto_data_series = self.df.groupby(asin_col)[sales_col].sum().sort_values(ascending=False).head(30)
#         pareto_data = {"labels": pareto_data_series.index.tolist(), "sales": pareto_data_series.values.tolist(), "cumulative_pct": (pareto_data_series.cumsum() / pareto_data_series.sum() * 100).round(1).tolist()}

#         print("--- 正在为每个季度计算明星产品矩阵数据 ---")
#         star_product_analysis = {}
#         first_listed_date_col = cols.get('first_listed_date')
#         listing_dates = pd.Series(dtype='datetime64[ns]')
#         if first_listed_date_col and first_listed_date_col in self.df.columns:
#             listing_dates = self.df.groupby(asin_col)[first_listed_date_col].first()

#         for p_type in product_types:
#             df_filtered = self.df if p_type == "Overall" else self.df[self.df[type_col] == p_type]
#             if df_filtered.empty: continue
#             df_filtered = df_filtered.copy()
#             df_filtered['quarter'] = df_filtered[date_col].dt.to_period('Q')
#             quarterly_sales = df_filtered.groupby([asin_col, 'quarter'])[sales_col].sum().unstack().fillna(0)

#             star_product_analysis[p_type] = {}
#             product_quadrant_paths = {}

#             for i in range(3, len(quarterly_sales.columns)):
#                 current_quarter = quarterly_sales.columns[i]
#                 recent_quarters = quarterly_sales.columns[i-3 : i+1]

#                 star_df = pd.DataFrame(quarterly_sales[recent_quarters].mean(axis=1), columns=['avg_sales_last_4q'])
#                 star_df = star_df[star_df['avg_sales_last_4q'] > 0]
#                 if star_df.empty: continue

#                 if not listing_dates.empty:
#                     star_df = star_df.join(listing_dates)

#                 points, mature_growth_scores = [], []
#                 today = current_quarter.end_time
#                 for asin, row in star_df.iterrows():
#                     is_new = False
#                     if first_listed_date_col in row and pd.notna(row[first_listed_date_col]):
#                         if (today - row[first_listed_date_col]).days <= 90: is_new = True

#                     points.append({"y": round(row['avg_sales_last_4q'], 0), "label": asin, "is_new": is_new, "sales_series": quarterly_sales.loc[asin, recent_quarters].values.tolist()})

#                 points_df_temp = pd.DataFrame(points)
#                 mature_products_df = points_df_temp[~points_df_temp['is_new']]

#                 for index, p in enumerate(points):
#                     if not p['is_new']:
#                         y_raw = np.array(p['sales_series'])
#                         y = np.log1p(winsorize(y_raw, limits=[0.05, 0.05]))
#                         x = np.arange(len(y))
#                         slopes = [(y[j] - y[i]) / (x[j] - x[i]) for i in range(len(y)) for j in range(i + 1, len(y)) if x[j] - x[i] != 0]
#                         growth_score = np.median(slopes) * 100 if slopes else 0
#                         mature_growth_scores.append(growth_score)
#                         points[index]['x'] = round(growth_score, 1)

#                 if mature_growth_scores:
#                     growth_p90 = np.percentile(mature_growth_scores, 90)
#                     growth_p65 = np.percentile(mature_growth_scores, 65)
#                     volume_p20 = mature_products_df['y'].quantile(0.20) if not mature_products_df.empty else 0
#                     for p in points:
#                         if p['is_new']:
#                             p['x'] = round(growth_p90 if p['y'] >= volume_p20 else growth_p65, 1)
#                             p['new_type'] = '高潜力新品' if p['y'] >= volume_p20 else '体量不足新品'

#                 for p in points:
#                     if 'sales_series' in p: del p['sales_series']

#                 points_df = pd.DataFrame([p for p in points if 'x' in p])

#                 baseline_available = False
#                 avg_sales, avg_growth = None, None
#                 mature_products_for_baseline = points_df[~points_df['is_new']]
#                 if len(mature_products_for_baseline) >= 5:
#                     baseline_available = True
#                     avg_sales = mature_products_for_baseline['y'].median()
#                     avg_growth = mature_products_for_baseline['x'].median()
#                     mean_sales = mature_products_for_baseline['y'].mean()
#                     mean_growth = mature_products_for_baseline['x'].mean()
#                     growth_p25 = mature_products_for_baseline['x'].quantile(0.25)
#                     growth_p75 = mature_products_for_baseline['x'].quantile(0.75)

#                 final_points = points_df.to_dict('records')
#                 if baseline_available:
#                     for p in final_points:
#                         if p['label'] not in product_quadrant_paths: product_quadrant_paths[p['label']] = []
#                         quadrant = ('high_growth' if p['x'] >= avg_growth else 'low_growth', 'high_sales' if p['y'] >= avg_sales else 'low_sales')
#                         product_quadrant_paths[p['label']].append(quadrant)

#                 star_product_analysis[p_type][str(current_quarter)] = {
#                     "points": final_points,
#                     "avg_sales": round(avg_sales, 2) if avg_sales is not None else None,
#                     "avg_growth": round(avg_growth, 2) if avg_growth is not None else None,
#                     "baseline_available": baseline_available,
#                     "mean_sales": round(mean_sales, 2) if mean_sales is not None else None,
#                     "mean_growth": round(mean_growth, 2) if mean_growth is not None else None,
#                     "growth_p25": round(growth_p25, 2) if avg_growth is not None else None,
#                     "growth_p75": round(growth_p75, 2) if avg_growth is not None else None
#                 }

#             stable_asins = set()
#             for asin, path in product_quadrant_paths.items():
#                 if len(path) > 1 and len(set(path)) == 1:
#                     stable_asins.add(asin)

#             for quarter_data in star_product_analysis[p_type].values():
#                 for point in quarter_data['points']:
#                     point['is_stable'] = point['label'] in stable_asins

#         print("--- 正在计算结构KPI时间线 (增量分解) ---")
#         structural_kpis = {}
#         first_listed_date_col = cols.get('first_listed_date')

#         for p_type in product_types:
#             df_filtered = self.df if p_type == "Overall" else self.df[self.df[type_col] == p_type]
#             if df_filtered.empty: continue

#             df_filtered = df_filtered.copy()
#             df_filtered['quarter'] = df_filtered[date_col].dt.to_period('Q')
#             all_quarters = sorted(df_filtered['quarter'].unique())

#             kpi_results = {
#                 'labels': [], 'top20_contrib': [], 'top50_contrib': [], 'new_sales_pct': [],
#                 'top20_inc_contrib': [], 'top50_inc_contrib': [], 'total_yoy': []
#             }

#             for i, quarter in enumerate(all_quarters):
#                 kpi_results['labels'].append(str(quarter))
#                 quarter_df = df_filtered[df_filtered['quarter'] == quarter]
#                 total_quarter_sales = quarter_df[sales_col].sum()

#                 sales_by_asin_current = quarter_df.groupby(asin_col)[sales_col].sum().sort_values(ascending=False)
#                 if total_quarter_sales > 0:
#                     kpi_results['top20_contrib'].append(round((sales_by_asin_current.head(20).sum() / total_quarter_sales) * 100, 1))
#                     kpi_results['top50_contrib'].append(round((sales_by_asin_current.head(50).sum() / total_quarter_sales) * 100, 1))
#                 else:
#                     kpi_results['top20_contrib'].append(0)
#                     kpi_results['top50_contrib'].append(0)

#                 top20_inc, top50_inc, total_yoy_val = None, None, None
#                 if i >= 4:
#                     prior_quarter_df = df_filtered[df_filtered['quarter'] == all_quarters[i-4]]
#                     total_prior_sales = prior_quarter_df[sales_col].sum()
#                     total_increment = total_quarter_sales - total_prior_sales

#                     if total_prior_sales > 0:
#                         total_yoy_val = round((total_increment / total_prior_sales) * 100, 1)

#                     if total_increment != 0:
#                         sales_by_asin_prior = prior_quarter_df.groupby(asin_col)[sales_col].sum()
#                         contrib_df = pd.DataFrame({'current': sales_by_asin_current, 'prior': sales_by_asin_prior}).fillna(0)
#                         contrib_df['increment'] = contrib_df['current'] - contrib_df['prior']
#                         top20_increment = contrib_df.loc[contrib_df.index.intersection(sales_by_asin_current.head(20).index)]['increment'].sum()
#                         top50_increment = contrib_df.loc[contrib_df.index.intersection(sales_by_asin_current.head(50).index)]['increment'].sum()
#                         top20_inc = round((top20_increment / total_increment) * 100, 1)
#                         top50_inc = round((top50_increment / total_increment) * 100, 1)

#                 kpi_results['top20_inc_contrib'].append(top20_inc)
#                 kpi_results['top50_inc_contrib'].append(top50_inc)
#                 kpi_results['total_yoy'].append(total_yoy_val)

#                 new_sales = 0
#                 if first_listed_date_col and first_listed_date_col in quarter_df.columns:
#                     quarter_end_date = quarter.end_time
#                     new_asins = quarter_df[(quarter_df[first_listed_date_col].notna()) & ((quarter_end_date - quarter_df[first_listed_date_col]).dt.days <= 90)][asin_col].unique()
#                     if len(new_asins) > 0:
#                         new_sales = quarter_df[quarter_df[asin_col].isin(new_asins)][sales_col].sum()
#                 kpi_results['new_sales_pct'].append(round((new_sales / total_quarter_sales) * 100, 1) if total_quarter_sales > 0 else 0)

#             structural_kpis[p_type] = kpi_results

#         print("--- 正在计算支持多维度的战略定位气泡图数据 ---")
#         strategic_positioning_data = {}

#         dims_to_analyze = {
#             'pricerange': cols.get('pricerange'),
#             'brand': cols.get('brand'),
#             'packsize': cols.get('packsize'),
#             'tiptype': cols.get('tiptype'),
#             'tiptype_packsize': (cols.get('tiptype'), cols.get('packsize'))
#         }

#         for p_type in product_types:
#             strategic_positioning_data[p_type] = {}
#             df_slice_by_type = self.df if p_type == "Overall" else self.df[self.df[type_col] == p_type]

#             for dim_key, dim_col_config in dims_to_analyze.items():
#                 is_multi_dim = isinstance(dim_col_config, tuple)
#                 group_cols = list(dim_col_config) if is_multi_dim else [dim_col_config]

#                 if any(c is None for c in group_cols) or not all(c in df_slice_by_type.columns for c in group_cols):
#                     continue

#                 if df_slice_by_type.empty: continue

#                 try:
#                     quarterly_total_sales_slice = df_slice_by_type.groupby(pd.Grouper(key=date_col, freq='Q'))[sales_col].sum()
#                     sales_by_dim = df_slice_by_type.groupby(group_cols + [pd.Grouper(key=date_col, freq='Q')])[sales_col].sum().unstack(level=date_col).fillna(0)

#                     if len(sales_by_dim.columns) < 8: continue

#                     recent_quarters = sales_by_dim.columns[-4:]
#                     prior_quarters = sales_by_dim.columns[-8:-4]

#                     if not all(q in quarterly_total_sales_slice.index for q in recent_quarters) or \
#                             not all(q in quarterly_total_sales_slice.index for q in prior_quarters):
#                         continue

#                     yearly_sales = sales_by_dim[recent_quarters].sum(axis=1)
#                     prior_yearly_sales = sales_by_dim[prior_quarters].sum(axis=1)

#                     total_market_yearly_sales = quarterly_total_sales_slice.loc[recent_quarters].sum()

#                     # 【核心修正】确保 market_share 和 yoy_growth 在多维交叉时能正确对齐
#                     market_share = (yearly_sales / total_market_yearly_sales) * 100 if total_market_yearly_sales > 0 else 0

#                     prior_yearly_sales_safe = prior_yearly_sales.where(prior_yearly_sales > 0, 1)
#                     yoy_growth = ((yearly_sales - prior_yearly_sales) / prior_yearly_sales_safe) * 100

#                     total_increment = quarterly_total_sales_slice.loc[recent_quarters].sum() - quarterly_total_sales_slice.loc[prior_quarters].sum()
#                     dim_increment = yearly_sales - prior_yearly_sales
#                     contrib_to_growth = (dim_increment / total_increment) * 100 if total_increment != 0 else 0

#                     bubble_data = []
#                     for dim_value_tuple in sales_by_dim.index:
#                         # 【核心修正】将复合索引元组转换为更美观的字符串
#                         label_str = str(dim_value_tuple) if not is_multi_dim else ' & '.join(map(str, dim_value_tuple))

#                         bubble_data.append({
#                             "label": label_str,
#                             "x": round(market_share.loc[dim_value_tuple], 1),
#                             "y": round(yoy_growth.loc[dim_value_tuple], 1),
#                             "r": int(yearly_sales.loc[dim_value_tuple]),
#                             "contrib": round(contrib_to_growth.loc[dim_value_tuple], 1)
#                         })

#                     strategic_positioning_data[p_type][dim_key] = bubble_data

#                 except Exception as e:
#                     print(f"❌ 为 '{p_type}' 的 '{dim_key}' 维度生成气泡图时发生错误: {e}")
#                     strategic_positioning_data[p_type][dim_key] = []

#         return {
#             "product_types": product_types, "time_events": dynamic_time_events, "salesForecast": forecast_data,
#             "quarterlyYoY": quarterly_yoy_data, "shareAnalysis": share_data, "growthTables": table_data,
#             "paretoAnalysis": pareto_data,
#             "starProductAnalysis": star_product_analysis,
#             "structuralKpis": structural_kpis,
#             "strategicPositioning": strategic_positioning_data
#         }

import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
import json
from datetime import datetime
from scipy.stats.mstats import winsorize
from itertools import combinations
import statsmodels.api as sm
import traceback

class SalesAnalyzer:
    """一个用于处理和分析销售数据的可复用、配置驱动的工具。
    专注于生成一个单一、全面、高度交互的HTML仪表板。(V9.6 稳定产品标识版)
    """

    def __init__(self, config: dict):
        self.config = config
        self.df = None
        print("✅ SalesAnalyzer 初始化成功。")

    def load_and_preprocess_data(self) -> bool:
        file_path = self.config.get('input_filepath')
        cols = self.config['columns']
        print(f"\n--- 正在从 '{file_path}' 加载数据 ---")
        try:
            self.df = pd.read_excel(file_path)
            print(f"  - 初始加载了 {len(self.df)} 行数据")
        except Exception as e:
            print(f"❌ {e}")
            traceback.print_exc()
            return False

        date_col = cols['date']
        try:
            self.df[date_col] = pd.to_datetime(self.df[date_col], format='%Y%m')
            print("日期成功按 'YYYYMM' 格式解析。")
        except (ValueError, TypeError):
            print("按 'YYYYMM' 格式解析失败，正在尝试自动识别标准日期格式...")
            self.df[date_col] = pd.to_datetime(self.df[date_col], errors='coerce')

        self.df.dropna(subset=[date_col], inplace=True)
        print(f"  - 清理无效日期后，剩下 {len(self.df)} 行数据")
        if not self.df.empty:
            unique_quarters = self.df[date_col].dt.to_period('Q').nunique()
            print(f"  - 剩余数据覆盖了 {unique_quarters} 个不重复的季度")

        sales_col = cols['sales']
        self.df[sales_col] = pd.to_numeric(self.df[sales_col], errors='coerce').fillna(0)
        self.df.sort_values(date_col, inplace=True)

        if self.df.empty:
            print("⚠️ 警告: 处理后没有剩下有效的数据。")
            return False

        first_listed_date_col = cols.get('first_listed_date')
        if first_listed_date_col and first_listed_date_col in self.df.columns:
            self.df[first_listed_date_col] = pd.to_datetime(self.df[first_listed_date_col], errors='coerce')

        print("✅ 数据加载与预处理成功。")
        return True

    def prepare_html_report_data(self) -> dict:
        """为交互式仪表板准备一个结构化的、JSON兼容的数据集合。"""
        print("\n--- 正在准备所有分析数据 ---")
        if self.df is None or self.df.empty:
            return {}

        cols = self.config['columns']
        date_col, sales_col, type_col, asin_col = cols['date'], cols['sales'], cols['type'], cols['asin']
        product_types = ["Overall"] + sorted(self.df[type_col].unique().tolist())

        dynamic_time_events = []
        year_agnostic_events = self.config.get("time_events", {})
        if not self.df.empty and year_agnostic_events:
            first_data_date = self.df[date_col].min()
            last_year_in_data = self.df[date_col].max().year
            for year in range(first_data_date.year, last_year_in_data + 2):
                for event_name, mm_dd in year_agnostic_events.items():
                    full_date_str = f"{year}-{mm_dd}"
                    try:
                        event_date = pd.to_datetime(full_date_str)
                        if event_date >= first_data_date:
                            dynamic_time_events.append({"label": event_name, "date": full_date_str})
                    except ValueError:
                        continue

        # --- 模块2：销售预测 (使用 SARIMAX) ---
        print("--- 正在计算销售预测 ---")
        forecast_data = {}
        for p_type in product_types:
            df_filtered = self.df if p_type == "Overall" else self.df[self.df[type_col] == p_type]
            monthly_sales = df_filtered.set_index(date_col)[sales_col].resample('M').sum()

            if len(monthly_sales) < 24:
                continue

            try:
                # SARIMA 模型代码
                model = SARIMAX(monthly_sales,
                                order=(1, 1, 1),
                                seasonal_order=(1, 1, 1, 12),
                                enforce_invertibility=False,
                                enforce_stationarity=False)

                model_fit = model.fit(disp=False)

                forecast_result = model_fit.get_forecast(steps=12)
                forecast_ci = forecast_result.conf_int()

                forecast_points = []
                for i in range(len(forecast_result.predicted_mean)):
                    date = forecast_result.predicted_mean.index[i].strftime('%Y-%m-%d')
                    value = forecast_result.predicted_mean.iloc[i]
                    lower = forecast_ci.iloc[i, 0]
                    upper = forecast_ci.iloc[i, 1]

                    forecast_points.append({
                        "x": date,
                        "y": round(value, 0),
                        "y_lower": round(lower, 0),
                        "y_upper": round(upper, 0)
                    })

                historical_points = [{'x': date.strftime('%Y-%m-%d'), 'y': value} for date, value in monthly_sales.items()]

                moving_average = monthly_sales.rolling(window=3).mean().dropna()
                moving_average_points = [{'x': date.strftime('%Y-%m-%d'), 'y': round(value, 0)} for date, value in moving_average.items()]

                all_data_series = pd.concat([monthly_sales, forecast_result.predicted_mean])
                x = np.arange(len(all_data_series))
                y = all_data_series.values
                z = np.polyfit(x, y, 1)
                p = np.poly1d(z)
                trend_line_points = [{'x': date.strftime('%Y-%m-%d'), 'y': round(p(i), 0)} for i, date in enumerate(all_data_series.index)]

                forecast_data[p_type] = {
                    "historical": historical_points,
                    "forecast": forecast_points,
                    "moving_average": moving_average_points,
                    "trend_line": trend_line_points
                }
            except Exception as e:
                print(f"❌ 为 '{p_type}' 类型生成 SARIMAX 预测时发生错误: {e}")
                traceback.print_exc()
                continue

        print("--- 正在计算季度同比增长 ---")
        sales_wide_q = self.df.groupby([pd.Grouper(key=date_col, freq='Q'), type_col])[sales_col].sum().unstack(type_col).fillna(0)
        yoy_wide_q = sales_wide_q.pct_change(periods=4) * 100
        quarterly_yoy_data = {"labels": sales_wide_q.index.to_period('Q').strftime('%YQ%q').tolist(), "sales_datasets": [{"label": str(col), "data": sales_wide_q[col].round(0).tolist()} for col in sales_wide_q.columns], "yoy_datasets": [{"label": str(col) + " YoY", "data": yoy_wide_q[col].where(pd.notna(yoy_wide_q[col]), None).round(1).tolist()} for col in yoy_wide_q.columns}]}

        print("--- 正在计算市场份额 ---")
        share_dimensions = ['type', 'brand', 'packsize', 'pricerange', 'tiptype']
        share_data = {}
        for dim in share_dimensions:
            dim_col_name = cols.get(dim)
            if not dim_col_name or dim_col_name not in self.df.columns: continue
            share_data[dim] = {}
            for p_type in product_types:
                df_filtered = self.df if p_type == "Overall" else self.df[self.df[type_col] == p_type]
                if df_filtered.empty: continue
                freq = 'M' if dim == 'type' else 'Q'
                time_format = '%Y-%m' if dim == 'type' else '%YQ%q'
                top_n = 20
                if dim == 'brand' and df_filtered[dim_col_name].nunique() > top_n:
                    total_sales = df_filtered.groupby(dim_col_name)[sales_col].sum()
                    top_brands = total_sales.nlargest(top_n).index.tolist()
                    df_with_others = df_filtered.copy()
                    df_with_others[dim_col_name] = df_with_others[dim_col_name].apply(lambda x: x if x in top_brands else '其他 (Others)')
                    data = df_with_others.groupby([pd.Grouper(key=date_col, freq=freq), dim_col_name])[sales_col].sum().unstack(dim_col_name).fillna(0)
                else:
                    data = df_filtered.groupby([pd.Grouper(key=date_col, freq=freq), dim_col_name])[sales_col].sum().unstack(dim_col_name).fillna(0)
                if data.empty: continue
                if '其他 (Others)' in data.columns:
                    other_col = data.pop('其他 (Others)')
                    data['其他 (Others)'] = other_col
                sorted_columns = data.sum().sort_values(ascending=False).index
                data_sorted = data[sorted_columns]
                data_sum = data_sorted.sum(axis=1)
                safe_data_sum = data_sum.where(data_sum != 0, 1)
                data_pct_sorted = data_sorted.div(safe_data_sum, axis=0) * 100
                datasets = [{"label": str(col), "data": data_pct_sorted[col].round(1).tolist(), "absoluteData": data_sorted[col].round(0).tolist()} for col in data_sorted.columns]
                share_data[dim][p_type] = { "labels": data_pct_sorted.index.to_period(freq).strftime(time_format).tolist(), "datasets": datasets }

        print("--- 正在计算增长表格 ---")
        table_dimensions = {'brand': ['brand'], 'packsize': ['packsize'], 'pricerange': ['pricerange'], 'tiptype': ['tiptype'], 'tiptype_packsize': ['tiptype', 'packsize']}
        table_data = {}
        for key, dim_names in table_dimensions.items():
            dim_cols = [cols.get(d) for d in dim_names if cols.get(d) in self.df.columns]
            if len(dim_cols) != len(dim_names): continue
            table_data[key] = {}
            for p_type in product_types:
                df_filtered = self.df if p_type == "Overall" else self.df[self.df[type_col] == p_type]
                if df_filtered.empty: continue
                sales = df_filtered.groupby(dim_cols + [pd.Grouper(key=date_col, freq='Q')])[sales_col].sum().unstack(date_col).fillna(0)
                if sales.empty: continue
                if len(sales.columns) > 0:
                    last_quarter_col = sales.columns[-1]
                    sales = sales.sort_values(by=last_quarter_col, ascending=False)
                if key == 'brand' and len(sales) > 20:
                    top_sales = sales.head(20)
                    other_sales = sales.iloc[20:].sum()
                    other_row = pd.DataFrame(other_sales).T
                    other_row.index = ['其他 (Others)']
                    sales = pd.concat([top_sales, other_row])
                if '其他 (Others)' in sales.index:
                    other_row = sales.loc[['其他 (Others)']]
                    sales = sales.drop('其他 (Others)')
                    sales = pd.concat([sales, other_row])
                display_map = self.config.get("header_mappings", {})
                display_names = [display_map.get(name.lower(), name.capitalize()) for name in dim_names]
                display_header = " & ".join(display_names)
                headers = [display_header] + [q.to_period('Q').strftime('%YQ%q') for q in sales.columns]
                rows = []
                for index, row_data in sales.iterrows():
                    row_content = [{'type': 'label', 'value': str(index)}]
                    for i, quarter in enumerate(sales.columns):
                        sale_val_num = row_data.get(quarter, 0)
                        yoy_val, yoy_status = None, 'neutral'
                        if sale_val_num == 0:
                            sale_val_str = "-"
                        else:
                            sale_val_str = f"{(sale_val_num / 10000.0):.2f}万"
                            if i >= 4:
                                prior_sale_val = row_data.get(sales.columns[i-4], 0)
                                if prior_sale_val > 0:
                                    yoy = (sale_val_num / prior_sale_val) - 1
                                    yoy_val, yoy_status = f"{yoy:.1%}", 'positive' if yoy > 0 else 'negative'
                                else:
                                    yoy_val, yoy_status = "New", 'positive'
                        row_content.append({'type': 'data', 'value': sale_val_str, 'yoy': yoy_val, 'yoy_status': yoy_status})
                    rows.append(row_content)
                table_data[key][p_type] = {"headers": headers, "rows": rows}

        print("--- 正在计算帕累托数据 ---")
        pareto_data_series = self.df.groupby(asin_col)[sales_col].sum().sort_values(ascending=False).head(30)
        pareto_data = {"labels": pareto_data_series.index.tolist(), "sales": pareto_data_series.values.tolist(), "cumulative_pct": (pareto_data_series.cumsum() / pareto_data_series.sum() * 100).round(1).tolist()}

        print("--- 正在为每个季度计算明星产品矩阵数据 ---")
        star_product_analysis = {}
        first_listed_date_col = cols.get('first_listed_date')
        listing_dates = pd.Series(dtype='datetime64[ns]')
        if first_listed_date_col and first_listed_date_col in self.df.columns:
            listing_dates = self.df.groupby(asin_col)[first_listed_date_col].first()

        for p_type in product_types:
            df_filtered = self.df if p_type == "Overall" else self.df[self.df[type_col] == p_type]
            if df_filtered.empty: continue
            df_filtered = df_filtered.copy()
            df_filtered['quarter'] = df_filtered[date_col].dt.to_period('Q')
            quarterly_sales = df_filtered.groupby([asin_col, 'quarter'])[sales_col].sum().unstack().fillna(0)

            star_product_analysis[p_type] = {}
            product_quadrant_paths = {}

            for i in range(3, len(quarterly_sales.columns)):
                current_quarter = quarterly_sales.columns[i]
                recent_quarters = quarterly_sales.columns[i-3 : i+1]

                star_df = pd.DataFrame(quarterly_sales[recent_quarters].mean(axis=1), columns=['avg_sales_last_4q'])
                star_df = star_df[star_df['avg_sales_last_4q'] > 0]
                if star_df.empty: continue

                if not listing_dates.empty:
                    star_df = star_df.join(listing_dates)

                points, mature_growth_scores = [], []
                today = current_quarter.end_time
                for asin, row in star_df.iterrows():
                    is_new = False
                    if first_listed_date_col in row and pd.notna(row[first_listed_date_col]):
                        if (today - row[first_listed_date_col]).days <= 90: is_new = True

                    points.append({"y": round(row['avg_sales_last_4q'], 0), "label": asin, "is_new": is_new, "sales_series": quarterly_sales.loc[asin, recent_quarters].values.tolist()})

                points_df_temp = pd.DataFrame(points)
                mature_products_df = points_df_temp[~points_df_temp['is_new']]

                for index, p in enumerate(points):
                    if not p['is_new']:
                        y_raw = np.array(p['sales_series'])
                        y = np.log1p(winsorize(y_raw, limits=[0.05, 0.05]))
                        x = np.arange(len(y))
                        slopes = [(y[j] - y[i]) / (x[j] - x[i]) for i in range(len(y)) for j in range(i + 1, len(y)) if x[j] - x[i] != 0]
                        growth_score = np.median(slopes) * 100 if slopes else 0
                        mature_growth_scores.append(growth_score)
                        points[index]['x'] = round(growth_score, 1)

                if mature_growth_scores:
                    growth_p90 = np.percentile(mature_growth_scores, 90)
                    growth_p65 = np.percentile(mature_growth_scores, 65)
                    volume_p20 = mature_products_df['y'].quantile(0.20) if not mature_products_df.empty else 0
                    for p in points:
                        if p['is_new']:
                            p['x'] = round(growth_p90 if p['y'] >= volume_p20 else growth_p65, 1)
                            p['new_type'] = '高潜力新品' if p['y'] >= volume_p20 else '体量不足新品'

                for p in points:
                    if 'sales_series' in p: del p['sales_series']

                points_df = pd.DataFrame([p for p in points if 'x' in p])

                baseline_available = False
                avg_sales, avg_growth = None, None
                mature_products_for_baseline = points_df[~points_df['is_new']]
                if len(mature_products_for_baseline) >= 5:
                    baseline_available = True
                    avg_sales = mature_products_for_baseline['y'].median()
                    avg_growth = mature_products_for_baseline['x'].median()
                    mean_sales = mature_products_for_baseline['y'].mean()
                    mean_growth = mature_products_for_baseline['x'].mean()
                    growth_p25 = mature_products_for_baseline['x'].quantile(0.25)
                    growth_p75 = mature_products_for_baseline['x'].quantile(0.75)

                final_points = points_df.to_dict('records')
                if baseline_available:
                    for p in final_points:
                        if p['label'] not in product_quadrant_paths: product_quadrant_paths[p['label']] = []
                        quadrant = ('high_growth' if p['x'] >= avg_growth else 'low_growth', 'high_sales' if p['y'] >= avg_sales else 'low_sales')
                        product_quadrant_paths[p['label']].append(quadrant)

                star_product_analysis[p_type][str(current_quarter)] = {
                    "points": final_points,
                    "avg_sales": round(avg_sales, 2) if avg_sales is not None else None,
                    "avg_growth": round(avg_growth, 2) if avg_growth is not None else None,
                    "baseline_available": baseline_available,
                    "mean_sales": round(mean_sales, 2) if mean_sales is not None else None,
                    "mean_growth": round(mean_growth, 2) if mean_growth is not None else None,
                    "growth_p25": round(growth_p25, 2) if avg_growth is not None else None,
                    "growth_p75": round(growth_p75, 2) if avg_growth is not None else None
                }

            stable_asins = set()
            for asin, path in product_quadrant_paths.items():
                if len(path) > 1 and len(set(path)) == 1:
                    stable_asins.add(asin)

            for quarter_data in star_product_analysis[p_type].values():
                for point in quarter_data['points']:
                    point['is_stable'] = point['label'] in stable_asins

        print("--- 正在计算结构KPI时间线 (增量分解) ---")
        structural_kpis = {}
        first_listed_date_col = cols.get('first_listed_date')

        for p_type in product_types:
            df_filtered = self.df if p_type == "Overall" else self.df[self.df[type_col] == p_type]
            if df_filtered.empty: continue

            df_filtered = df_filtered.copy()
            df_filtered['quarter'] = df_filtered[date_col].dt.to_period('Q')
            all_quarters = sorted(df_filtered['quarter'].unique())

            kpi_results = {
                'labels': [], 'top20_contrib': [], 'top50_contrib': [], 'new_sales_pct': [],
                'top20_inc_contrib': [], 'top50_inc_contrib': [], 'total_yoy': []
            }

            for i, quarter in enumerate(all_quarters):
                kpi_results['labels'].append(str(quarter))
                quarter_df = df_filtered[df_filtered['quarter'] == quarter]
                total_quarter_sales = quarter_df[sales_col].sum()

                sales_by_asin_current = quarter_df.groupby(asin_col)[sales_col].sum().sort_values(ascending=False)
                if total_quarter_sales > 0:
                    kpi_results['top20_contrib'].append(round((sales_by_asin_current.head(20).sum() / total_quarter_sales) * 100, 1))
                    kpi_results['top50_contrib'].append(round((sales_by_asin_current.head(50).sum() / total_quarter_sales) * 100, 1))
                else:
                    kpi_results['top20_contrib'].append(0)
                    kpi_results['top50_contrib'].append(0)

                top20_inc, top50_inc, total_yoy_val = None, None, None
                if i >= 4:
                    prior_quarter_df = df_filtered[df_filtered['quarter'] == all_quarters[i-4]]
                    total_prior_sales = prior_quarter_df[sales_col].sum()
                    total_increment = total_quarter_sales - total_prior_sales

                    if total_prior_sales > 0:
                        total_yoy_val = round((total_increment / total_prior_sales) * 100, 1)

                    if total_increment != 0:
                        sales_by_asin_prior = prior_quarter_df.groupby(asin_col)[sales_col].sum()
                        contrib_df = pd.DataFrame({'current': sales_by_asin_current, 'prior': sales_by_asin_prior}).fillna(0)
                        contrib_df['increment'] = contrib_df['current'] - contrib_df['prior']
                        top20_increment = contrib_df.loc[contrib_df.index.intersection(sales_by_asin_current.head(20).index)]['increment'].sum()
                        top50_increment = contrib_df.loc[contrib_df.index.intersection(sales_by_asin_current.head(50).index)]['increment'].sum()
                        top20_inc = round((top20_increment / total_increment) * 100, 1)
                        top50_inc = round((top50_increment / total_increment) * 100, 1)

                kpi_results['top20_inc_contrib'].append(top20_inc)
                kpi_results['top50_inc_contrib'].append(top50_inc)
                kpi_results['total_yoy'].append(total_yoy_val)

                new_sales = 0
                if first_listed_date_col and first_listed_date_col in quarter_df.columns:
                    quarter_end_date = quarter.end_time
                    new_asins = quarter_df[(quarter_df[first_listed_date_col].notna()) & ((quarter_end_date - quarter_df[first_listed_date_col]).dt.days <= 90)][asin_col].unique()
                    if len(new_asins) > 0:
                        new_sales = quarter_df[quarter_df[asin_col].isin(new_asins)][sales_col].sum()
                kpi_results['new_sales_pct'].append(round((new_sales / total_quarter_sales) * 100, 1) if total_quarter_sales > 0 else 0)

            structural_kpis[p_type] = kpi_results

        print("--- 正在计算支持多维度的战略定位气泡图数据 ---")
        strategic_positioning_data = {}

        dims_to_analyze = {
            'pricerange': cols.get('pricerange'),
            'brand': cols.get('brand'),
            'packsize': cols.get('packsize'),
            'tiptype': cols.get('tiptype'),
            'tiptype_packsize': (cols.get('tiptype'), cols.get('packsize'))
        }

        for p_type in product_types:
            strategic_positioning_data[p_type] = {}
            df_slice_by_type = self.df if p_type == "Overall" else self.df[self.df[type_col] == p_type]

            for dim_key, dim_col_config in dims_to_analyze.items():
                is_multi_dim = isinstance(dim_col_config, tuple)
                group_cols = list(dim_col_config) if is_multi_dim else [dim_col_config]

                if any(c is None for c in group_cols) or not all(c in df_slice_by_type.columns for c in group_cols):
                    continue

                if df_slice_by_type.empty: continue

                try:
                    quarterly_total_sales_slice = df_slice_by_type.groupby(pd.Grouper(key=date_col, freq='Q'))[sales_col].sum()
                    sales_by_dim = df_slice_by_type.groupby(group_cols + [pd.Grouper(key=date_col, freq='Q')])[sales_col].sum().unstack(level=date_col).fillna(0)

                    if len(sales_by_dim.columns) < 8: continue

                    recent_quarters = sales_by_dim.columns[-4:]
                    prior_quarters = sales_by_dim.columns[-8:-4]

                    if not all(q in quarterly_total_sales_slice.index for q in recent_quarters) or \
                            not all(q in quarterly_total_sales_slice.index for q in prior_quarters):
                        continue

                    yearly_sales = sales_by_dim[recent_quarters].sum(axis=1)
                    prior_yearly_sales = sales_by_dim[prior_quarters].sum(axis=1)

                    total_market_yearly_sales = quarterly_total_sales_slice.loc[recent_quarters].sum()

                    # 【核心修正】确保 market_share 和 yoy_growth 在多维交叉时能正确对齐
                    market_share = (yearly_sales / total_market_yearly_sales) * 100 if total_market_yearly_sales > 0 else 0

                    prior_yearly_sales_safe = prior_yearly_sales.where(prior_yearly_sales > 0, 1)
                    yoy_growth = ((yearly_sales - prior_yearly_sales) / prior_yearly_sales_safe) * 100

                    total_increment = quarterly_total_sales_slice.loc[recent_quarters].sum() - quarterly_total_sales_slice.loc[prior_quarters].sum()
                    dim_increment = yearly_sales - prior_yearly_sales
                    contrib_to_growth = (dim_increment / total_increment) * 100 if total_increment != 0 else 0

                    bubble_data = []
                    for dim_value_tuple in sales_by_dim.index:
                        # 【核心修正】将复合索引元组转换为更美观的字符串
                        label_str = str(dim_value_tuple) if not is_multi_dim else ' & '.join(map(str, dim_value_tuple))

                        bubble_data.append({
                            "label": label_str,
                            "x": round(market_share.loc[dim_value_tuple], 1),
                            "y": round(yoy_growth.loc[dim_value_tuple], 1),
                            "r": int(yearly_sales.loc[dim_value_tuple]),
                            "contrib": round(contrib_to_growth.loc[dim_value_tuple], 1)
                        })

                    strategic_positioning_data[p_type][dim_key] = bubble_data

                except Exception as e:
                    print(f"❌ 为 '{p_type}' 的 '{dim_key}' 维度生成气泡图时发生错误: {e}")
                    traceback.print_exc()
                    strategic_positioning_data[p_type][dim_key] = []

        return {
            "product_types": product_types, "time_events": dynamic_time_events, "salesForecast": forecast_data,
            "quarterlyYoY": quarterly_yoy_data, "shareAnalysis": share_data, "growthTables": table_data,
            "paretoAnalysis": pareto_data,
            "starProductAnalysis": star_product_analysis,
            "structuralKpis": structural_kpis,
            "strategicPositioning": strategic_positioning_data
        }







