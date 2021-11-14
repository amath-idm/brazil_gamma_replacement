'''
Load and analyze vaccine and case data
'''

#%% Imports
import numpy as np
import pandas as pd
import sciris as sc
import pylab as pl
import statsmodels.formula.api as smfa
import scipy.stats as st

#%% Settings
do_plot = 1
do_show = 1
do_save = 1
post_vx = 14 # Number of days post-vaccine to start counting
count_both = False # Count only people who have received both doses


#%% Load data
sc.heading('Loading data...')
raw = sc.objdict() # For storing all data
d = sc.objdict() # For storing processed data
csdkeys = ['cases', 'severe', 'deaths']
allkeys = csdkeys + ['vaccines']
datadir = sc.thisdir(aspath=True) / 'data'
for key in allkeys:
    filename = datadir / (key + '.xlsx')
    print(f'Loading "{filename}" as "{key}"')
    raw[key] = pd.read_excel(filename) # Load data
raw.vx_data = pd.read_excel(filename, sheet_name='paper_data_new')


#%% Process data
sc.heading('Processing data...')

# Copy into new structure
for key in raw.keys():
    d[key] = sc.dcp(raw[key])

# Rename columns from e.g.  '75 A 79' to '75' and drop missing values
start_day = '2020-01-01'

for key in csdkeys:
    cols = d[key].columns
    mapping = {k:k.split()[0] for k in cols}
    d[key] = d[key].rename(columns=mapping)
    d[key] = d[key].dropna()

# Add a "day" column
for key in allkeys:
    d[key]['day'] = sc.day(sc.readdate(d[key].date.tolist()), start_day=start_day)


# Age calculations
bins = list(d.cases.columns)
for key in ['date', 'day', 'total', 'total_under65', 'total_under60']:
    bins.remove(key)
bins.append('100')
bins = np.array([int(b) for b in bins])
bin_start = bins[:-1]
bin_end = bins[1:]

# Calculate vaccine coverage
coverage = {}
coverage_low = {}
coverage_high = {}
both = {}
both_low = {}
both_high = {}
coverage_start = {}
coverage_end = {}
for i,row in d.vaccines.iterrows():
    if np.isfinite(row.age_doses):
        age = row.age_bin
        ind = sc.findfirst(bin_start == age)
        if age not in coverage_start:
            coverage_start[age] = row.day
        coverage_end[age] = row.day

for i,row in d.vx_data.iterrows():
    age = row.age_bin
    coverage[age] = min(1, row.cov_mean)
    both[age] = min(1, row.both_dose_mean)
    old = min(1, row.old_pop_new_cov)
    new = min(1, row.old_pop_new_cov)
    coverage_low[age] = min(old, new)
    coverage_high[age] = max(old, new)

if count_both:
    coverage = both
print('Final coverage values:')
for age in coverage.keys():
    print(f'Age {age} = {coverage[age]} ({coverage_low[age]}-{coverage_high[age]})')


#%% Analysis
sc.heading('Performing analysis...')

n = len(coverage) # How many data points we have
res = sc.objdict()
res.covbins = np.array(list(coverage.keys()))
order = np.argsort(res.covbins)
res.covbins = res.covbins[order]
res.coverage = np.array([coverage[k] for k in res.covbins])
res.both = np.array([both[k] for k in res.covbins])
res.slope = sc.objdict()
res.slope_low = sc.objdict()
res.slope_high = sc.objdict()
res.bslope = sc.objdict()
res.bslope_low = sc.objdict()
res.bslope_high = sc.objdict()
res.cases, res.severe, res.deaths = [], [], []
res.rho = sc.objdict()
res.p = sc.objdict()
res.brho = sc.objdict()
res.bp = sc.objdict()
for key in csdkeys:
    res[key] = sc.objdict()
    for subkey in ['tot_before', 'tot_after', 'agebin_before', 'agebin_after']:
        res[key][subkey] = np.zeros(n)

    for i,cb in enumerate(res.covbins):
        strkey = str(int(cb)) # Convert e.g. 90.0 to '90'
        before = sc.findinds(d[key].day < coverage_start[cb])
        after = sc.findinds(d[key].day > (coverage_end[cb] + post_vx))
        res[key].tot_before[i]    = d[key].total_under60[before].sum()
        res[key].tot_after[i]     = d[key].total_under60[after].sum()
        res[key].agebin_before[i] = d[key][strkey][before].sum() # e.g. d.cases['30'][:417].sum()
        res[key].agebin_after[i]  = d[key][strkey][after].sum()

    res[key].frac_before = res[key].agebin_before/res[key].tot_before
    res[key].frac_after  = res[key].agebin_after/res[key].tot_after
    res[key].ratio = res[key].frac_after/res[key].frac_before
    res[key].change = (res[key].ratio - 1)*100

    # Calculate slope and uncertainties
    data = pd.DataFrame(dict(x=res.coverage, y=res[key].change/100))
    data = data.append(dict(x=0, y=0), ignore_index=True)
    fit = smfa.ols(formula="y ~ x - 1", data=data).fit()
    conf = fit.conf_int(alpha=0.05, cols=None)
    res.slope[key] = fit.params[0]
    res.slope_low[key]  = conf[0].values[0]
    res.slope_high[key] = conf[1].values[0]
    rho, p = st.pearsonr(data.x, data.y)
    res.rho[key] = rho
    res.p[key] = p

    # And again, for both doses
    bdata = pd.DataFrame(dict(x=res.both, y=res[key].change/100))
    bdata = bdata.append(dict(x=0, y=0), ignore_index=True)
    bfit = smfa.ols(formula="y ~ x - 1", data=bdata).fit()
    bconf = bfit.conf_int(alpha=0.05, cols=None)
    res.bslope[key] = bfit.params[0]
    res.bslope_low[key]  = bconf[0].values[0]
    res.bslope_high[key] = bconf[1].values[0]
    rho, p = st.pearsonr(bdata.x, bdata.y)
    res.brho[key] = rho
    res.bp[key] = p


print('Results:')
dfs = sc.objdict()
for key in csdkeys:
    df = pd.DataFrame(res[key])
    df.index = res.covbins
    print(df)
    dfs[key] = df


#%% Plotting

if do_plot:
    sc.heading('Plotting...')

    pl.rc('figure', dpi=150)
    fig = pl.figure(figsize=(8,10))
    pl.subplots_adjust(left=0.13, right=0.95, bottom=0.05, top=0.95, hspace=0.3)

    # Figure text
    x1 = 0.02
    y1 = 0.98
    dy = 0.33
    fsize = 24
    pl.figtext(x1, 0.96, 'a', fontsize=fsize)
    pl.figtext(x1, 0.65, 'b', fontsize=fsize)
    pl.figtext(x1, 0.31, 'c', fontsize=fsize)

    axs = sc.objdict()
    axs.cases  = pl.subplot(3,1,1)
    axs.severe = pl.subplot(3,1,2)
    axs.deaths = pl.subplot(3,1,3)

    xlims = np.array([0,100])
    covlabels = []
    covbins = np.int64(res.covbins)
    for i in range(len(covbins)-1):
        covlabels.append(f'Age {covbins[i]}–{covbins[i+1]-1}') # ({res.coverage[i]*100:0.0f}%)
    covlabels.append(f'Age {covbins[-1]}+') #  ({res.coverage[-1]*100:0.0f}%)

    colors = sc.vectocolor(res.covbins, reverse=True)
    for key in csdkeys:
        ax = axs[key]

        # Plot 1:1
        plot_1_1 = False
        if plot_1_1:
            ax.plot(xlims, -xlims, '--', lw=1, alpha=0.4, c='k')

        # Plot points
        for i in range(n):
            label = covlabels[i] if key == 'cases' else None
            blabel = '2 doses'if key == 'cases' and i == n-1 else None
            rcov = res.coverage[i]*100
            rbo = res.both[i]*100
            rch = res[key].change[i]
            col = colors[i]
            ax.plot(rcov, rch, 'o', c=col, label=label, markersize=8, zorder=10)
            ax.plot(rbo, rch, 'd', c=col, label=blabel, alpha=0.5, markersize=6, zorder=9)
            ax.plot([rcov, rbo], [rch]*2, '--', c=col, lw=1, alpha=0.5, zorder=8)

        # Plot fit
        slope = res.slope[key]
        slope_low = res.slope_low[key]
        slope_high = res.slope_high[key]
        slopelabel = f'Slope, ≥1 dose: {slope:0.2f} (95% CI: {slope_low:0.2f}, {slope_high:0.2f})'
        ax.plot(xlims, slope*xlims, label=slopelabel)

        # Plot both dose fit
        bslope = res.bslope[key]
        bslope_low = res.bslope_low[key]
        bslope_high = res.bslope_high[key]
        bslopelabel = f'Slope, 2 doses: {bslope:0.2f} (95% CI: {bslope_low:0.2f}, {bslope_high:0.2f})'
        ax.plot(xlims, bslope*xlims, label=bslopelabel)

        # Plot uncertainty
        y1 = slope_low*xlims
        y2 = slope_high*xlims
        ax.fill_between(xlims, y1, y2, zorder=-10, alpha=0.1)

        # Plot both dose uncertainty
        y1 = bslope_low*xlims
        y2 = bslope_high*xlims
        ax.fill_between(xlims, y1, y2, zorder=-10, alpha=0.1)

        # Other tidying
        ax.legend(frameon=False, loc='lower left')
        ylabkey = key if key != 'severe' else key + ' cases'
        ax.set_ylabel(f'Change in fraction of {ylabkey} (%)')
        if key == 'deaths':
            ax.set_xlabel('Vaccine coverage (%)')
        ax.set_ylim([-100, 0])
        ax.set_xlim([0,101])
        sc.boxoff(ax)

    if do_save:
        pl.savefig('final_vaccine_efficacy.tif', dpi=600, format="tiff", pil_kwargs={"compression": "tiff_lzw"})
        pl.savefig('final_vaccine_efficacy.png', dpi=300)
        pl.savefig('final_vaccine_efficacy.pdf', dpi=300)
    if do_show:
        pl.show()


r = res.slope
rl = res.slope_low
rh = res.slope_high
br = res.bslope
brl = res.bslope_low
brh = res.bslope_high
text = f'''\
As shown in Figure 6, vaccination was associated with a moderate reduction in the number of cases
(best-fit slope {r.cases:0.2f}, 95% CI: {rh.cases:0.2f}, {rl.cases:0.2f} for ≥1 dose;
 best-fit slope {br.cases:0.2f}, 95% CI: {brh.cases:0.2f}, {brl.cases:0.2f} for 2 doses).
However, it was associated with a pronounced reduction in severe cases
({r.severe:0.2f}, 95% CI: {rh.severe:0.2f}, {rl.severe:0.2f} for ≥1 dose;
 {br.severe:0.2f}, 95% CI: {brh.severe:0.2f}, {brl.severe:0.2f} for 2 doses)
and deaths
({r.deaths:0.2f}, 95% CI: {rh.deaths:0.2f}, {rl.deaths:0.2f} for ≥1 dose;
 {br.deaths:0.2f}, 95% CI: {brh.deaths:0.2f}, {brl.deaths:0.2f} for 2 doses).
'''

print(text.replace('\n', ' ').replace('-', '–').replace('  ', ' '))

print('Done')


