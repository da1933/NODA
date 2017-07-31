Code = pd.read_stata("source/code-cln.dta")
REASCD = Code[Code['code_type'] == 'REASCD']
merged_cat_filled['DFDN_STATUS_REASON_CODE'] =\
merged_cat_filled['DFDN_STATUS_REASON_CODE'].astype(str)
merged_cat_filled['DFDN_STATUS_REASON_CODE'] = \
merged_cat_filled['DFDN_STATUS_REASON_CODE'].str.split(".").apply(lambda x: x[0])
merged_cat_filled.DFDN_STATUS_REASON_CODE =\
merged_cat_filled.DFDN_STATUS_REASON_CODE.apply(lambda x: x.zfill(3))

REASCD['code_code'] = REASCD['code_code'].astype(str)

merged_cat_filled = pd.merge(merged_cat_filled, REASCD[['code_code', 'short_desc', 'long_desc']], \
                             left_on=['DFDN_STATUS_REASON_CODE'], right_on=['code_code'], how='left')
