'''
Load and analyze vaccine and case data
'''

#%% Imports
import numpy as np
import pandas as pd
import sciris as sc
import pylab as pl
import covasim as cv
import statsmodels.formula.api as smfa

#%% Settings
do_plot = 1
do_show = 1
do_save = 1
post_vx = 14 # Number of days post-vaccine to start counting


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
popfile = datadir / 'population.xlsx'
rawpop = pd.read_excel(popfile)


#%% Process data
sc.heading('Processing data...')

# Copy into new structure
for key in allkeys:
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

# Determine age distributions
pop_size = 464983 # Wikipedia article on "São José do Rio Preto"
age_dist = cv.data.get_age_distribution(location='brazil')
bins = list(d.cases.columns)
for key in ['date', 'day', 'total', 'total_under65']:
    bins.remove(key)
bins.append('100')
bins = np.array([int(b) for b in bins])
bin_start = bins[:-1]
bin_end = bins[1:]
bin_size = bin_end - bin_start
n_bins = len(bin_start)
pop_dist = np.zeros(n_bins)
for i,bs in enumerate(bin_start):
    bin_ind = sc.findlast(age_dist[:,0] <= bs)
    prop = age_dist[bin_ind,2]
    bin_width = age_dist[bin_ind,1] - age_dist[bin_ind,0] + 1 # +1 since listed as e.g. [40,49] instead of [40,50]
    pop_dist[i] = pop_size * prop * bin_size[i]/bin_width
assert np.isclose(pop_dist.sum(), pop_size) # Check our math is right, before modifying
pop_dist[-1] = 2510/2 # From the vaccine spreadsheet, saying 2510 doses = 100% coverage for 90+
pop_dist[-2] = 3031/0.88/2 # From the vaccine spreadsheet, saying 3031 doses = 88% coverage for 85-89
pop_dist[-3] = 5374/0.78/2 # From the vaccine spreadsheet, saying 5374 doses = 78% coverage for 80-84
pop_dist[-4] = 2010/0.37/2 + 4620/0.9/2 # From the vaccine spreadsheet, saying 2010 doses = 37% coverage for 77-79
pop_dist = pop_dist.round() # Don't keep fractional people
orig_pop_dist = sc.dcp(pop_dist)
pop_dist = rawpop['População'].values


# Calculate vaccine coverage
coverage = {}
coverage_start = {}
coverage_end = {}
for i,row in d.vaccines.iterrows():
    if np.isfinite(row.age_doses):
        age = row.age_bin
        ind = sc.findfirst(bin_start == age)
        cov = row.age_doses/pop_dist[ind]/2 # Since need 2 doses
        if age not in coverage:
            coverage[age] = 0
            coverage_start[age] = row.day
        coverage[age] += cov
        coverage_end[age] = row.day
        print(f'On day {row.day}, {row.age_doses} doses for {pop_dist[ind]} people aged {age} increases coverage to {coverage[age]}')

print('Final coverage values:')
print(coverage)
print('Warning: coverage for 75-79 looks too low, check source data!')


#%% Analysis
sc.heading('Performing analysis...')

n = len(coverage) # How many data points we have
res = sc.objdict()
res.covbins = np.array(list(coverage.keys()))
order = np.argsort(res.covbins)
res.covbins = res.covbins[order]
res.coverage = np.array([coverage[k] for k in res.covbins])
res.slope = sc.objdict()
res.slope_best = sc.objdict()
res.slope_low = sc.objdict()
res.slope_high = sc.objdict()
res.cases, res.severe, res.deaths = [], [], []
for key in csdkeys:
    res[key] = sc.objdict()
    for subkey in ['tot_before', 'tot_after', 'agebin_before', 'agebin_after']:
        res[key][subkey] = np.zeros(n)

    for i,cb in enumerate(res.covbins):
        strkey = str(int(cb)) # Convert e.g. 90.0 to '90'
        before = sc.findinds(d[key].day < coverage_start[cb])
        after = sc.findinds(d[key].day > (coverage_end[cb] + post_vx))
        res[key].tot_before[i]    = d[key].total_under65[before].sum()
        res[key].tot_after[i]     = d[key].total_under65[after].sum()
        res[key].agebin_before[i] = d[key][strkey][before].sum() # e.g. d.cases['30'][:417].sum()
        res[key].agebin_after[i]  = d[key][strkey][after].sum()

    res[key].frac_before = res[key].agebin_before/res[key].tot_before
    res[key].frac_after  = res[key].agebin_after/res[key].tot_after
    res[key].ratio = res[key].frac_after/res[key].frac_before
    res[key].change = (res[key].ratio - 1)*100

    # Calculate slope and uncertainties
    data = pd.DataFrame(dict(x=res.coverage, y=res[key].change/100))
    fit = smfa.ols(formula="y ~ x - 1", data=data).fit()
    conf = fit.conf_int(alpha=0.05, cols=None)
    res.slope[key] = fit.params[0]
    res.slope_low[key]  = conf[0].values[0]
    res.slope_high[key] = conf[1].values[0]

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
        covlabels.append(f'Ages {covbins[i]}–{covbins[i+1]-1}') # ({res.coverage[i]*100:0.0f}%)
    covlabels.append(f'Ages {covbins[-1]}+') #  ({res.coverage[-1]*100:0.0f}%)

    colors = sc.vectocolor(res.covbins, reverse=True)
    for key in csdkeys:
        ax = axs[key]

        # Plot 1:1
        ax.plot(xlims, -xlims, '--', lw=1, alpha=0.4, c='k')

        # Plot points
        for i in range(n):
            label = covlabels[i] if key == 'cases' else None
            ax.plot(res.coverage[i]*100, res[key].change[i], 'o', c=colors[i], label=label)

        # Plot fit
        slope = res.slope[key]
        slope_low = res.slope_low[key]
        slope_high = res.slope_high[key]
        slopelabel = f'Slope: {slope:0.2f} (95% CI: {slope_low:0.2f}, {slope_high:0.2f})'
        ax.plot(xlims, slope*xlims, label=slopelabel)

        # Plot uncertainty
        y1 = slope_low*xlims
        y2 = slope_high*xlims
        ax.fill_between(xlims, y1, y2, zorder=-10, alpha=0.1)

        # Other tidying
        ax.legend(frameon=False, loc='lower left')
        ylabkey = key if key != 'severe' else key + ' cases'
        ax.set_ylabel(f'Change in fraction of {ylabkey} (%)')
        if key == 'deaths':
            ax.set_xlabel('Vaccine coverage (%)')
        ax.set_ylim([-100, 0])
        ax.set_xlim(xlims)
        sc.boxoff(ax)

    if do_save:
        cv.savefig('vaccine_efficacy.png')
    if do_show:
        pl.show()


print('Done')


