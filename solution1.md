# Solution 1: funtional HEP

## Background

One Higgs boson decays into two Z boson. One Z boson decays into two muons, or two electrons. These processes happen so quickly that we can only detect the resulting muons/electrons. Higgs bosons or Z bosons can be discovered only through particles that they have decayed into. We use the fact that masses of Higgs bosons and Z bosons are somewhat well defined, and that energy and momentum are conserved. So they can be calculated from total energy and momentum of their decay products as follow: 
![](https://snag.gy/epP9kD.jpg)

There can be a number of particles captured in a single event of collision. To know which one decays from Z bosons, we can calculate mass of every pairs of muons (as well as electrons). By drawing a histogram, mass of Z bosons will show as a peak.

The data contains energy and momentum of muons and electrons created in a number of events. To discover Higgs bosons from the given information, this solution applied two different appraches.


## Higgs bosons to muons and/or electrons

We know that one Higgs decays into two Z bosons. Each of Z bosons decays into either a pair of muons, or a pair of electrons. The product can then be: (1) two pairs of electrons, (2) two pairs of muons and (3) a pair of muons and a pair of electrons. 
First we can get a list of all muons and electron pairs, among which some compose Z bosons. Without identifying intermediate Z bosons, we pair up those pairs to compute Higgs bosons mass.

```python
def higgs_mass_pairs(muons, electrons):
	all_pairs = muons.pairs(lambda x ,y: (x ,y)) 
	el = electrons.pairs(lambda x, y: (x ,y))
	all_pairs.extend(el)
	return all_pairs.pairs(lambda x, y: mass(*(x+y)))
```

Since the number of muons and electrons from a Higgs boson is at least four, we can filter out event whose sum of muons and electrons is less than that. We then compute mass of original particles and plot a histogram in order to identify Higgs bosons.

```python
masses = (events
	.filter(lambda event: event.muons.size + event.electrons.size >= 4)
	.map(lambda event: higgs_mass_pairs(event.muons, event.electrons)))
```

![](https://snag.gy/2ISPVD.jpg)


## Higgs bozons to Z bosons, Z bosons to muons and/or electrons

Another way to discover Higgs bosons is to first identify Z bosons from muons and electrons. Then we can find Higgs bosons from those Z bosons in the same manner.

In this approach, total energy, momentum and mass of each muon pairs and electron pairs are computed. Masses are used directly to create a histogram in order to identify the peak range. Their summed energy and momentum will be used later to compute mass of Higgs bosons. All three values returned as a tuple (denoted by *mep*) to avoid redundant computation.

```python
def z_mep_pairs(muons, electrons):
	all_pairs = muons.pairs(mass_energy_momentum)
	el = electrons.pairs(mass_energy_momentum)
	all_pairs.extend(el)
	return all_pairs	# return [mass, energy, momentum] of every pair

mue_pairs_mep = (events
		.filter(lambda event: event.muons.size + event.electrons.size >= 4) 
		.map(lambda event: z_mep_pairs(event.muons, event.electrons)))
```


To identify Z bosons, histogram is constructed from masses of those pairs. The peak indicates Z bosons, whose mass range is then returned for later use to select muon/electron pairs

```python
def histogram(data, bin_width=5):
	dat_min, dat_max = data.reduce(get_min_max)
	dat_min = dat_min - (dat_min % bin_width)
	hist_range = dat_max - dat_min
	hist_size = int((hist_range + bin_width) / bin_width)
	intervals = range(hist_size).map(lambda x: dat_min + x * bin_width)
	hist = list(np.zeros(hist_size, dtype=(int)))
	for dat in data:
		ind = int((dat - dat_min) / bin_width)
		hist[ind] = hist[ind] + 1

	peak = hist.index(max(hist)) * bin_width + dat_min
	return hist, (peak, peak+bin_width), intervals


mue_pairs_masses = mue_pairs_mep.flatten.map(lambda pair: pair[0])
z_hist, (z_peak_min, z_peak_max), _ = histogram(mue_pairs_masses)

```


From histogram peak range, we can select only pairs that compose Z bosons. Similar to the previous approach, events where the number Z boson pairs is less than two can be filtered out. 

```python
z_pairs_mep = (mue_pairs_mep
		.map(lambda event: event
			.filter(lambda pair: pair[0] >= z_peak_min and pair[0] <= z_peak_max))
		.filter(lambda event: event.size >= 2))
```

Again, mass of a Z boson pair is calculated from energy and momentum to find a mass of their original particle. Lastly, we can identify a Higgs boson mass range using a histogram.

```python
higgs_masses = (z_pairs_mep.map(lambda event: event.pairs(lambda x, y: mass_from_mep(x,y))))

h_hist, (h_peak_min, h_peak_max), h_intv = histogram(higgs_masses.flatten, 10)
```

![](https://snag.gy/PtiGTL.jpg)

The last two parts can be combined to one chain functional call. Although it is possible to apply reduce on the list to get Z mass range, we would need to recalculate the *mep* values again so it is done off the chain here.

```python
mue_pairs_mep = (events
		.filter(lambda event: event.muons.size + event.electrons.size >= 4) 
		.map(lambda event: z_mep_pairs(event.muons, event.electrons)))

z_hist, (z_peak_min, z_peak_max), _ = histogram(mue_pairs_mep
	.flatten.map(lambda pair: pair[0]))

higgs_masses = (mue_pairs_mep
		.map(lambda event: event
			.filter(lambda pair: pair[0] >= z_peak_min and pair[0] <= z_peak_max))
		.filter(lambda event: event.size >= 2)
		.map(lambda event: event.pairs(lambda x, y: mass_from_mep(x,y))))
```