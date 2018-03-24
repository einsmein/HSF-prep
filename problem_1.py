import oamap.source.root
import uproot
import functional
from math import *
import matplotlib.pyplot as plt
import numpy as np

events = uproot.open("http://scikit-hep.org/uproot/examples/HZZ.root")["events"].oamap()
events.schema.content.rename("NElectron", "electrons")
events.schema.content["electrons"].content.rename("Electron_Px", "px")
events.schema.content["electrons"].content.rename("Electron_Py", "py")
events.schema.content["electrons"].content.rename("Electron_Pz", "pz")
events.schema.content["electrons"].content.rename("Electron_E", "energy")
events.schema.content["electrons"].content.rename("Electron_Iso", "isolation")
events.schema.content["electrons"].content.rename("Electron_Charge", "charge")
events.schema.content.rename("NMuon", "muons")
events.schema.content["muons"].content.rename("Muon_Px", "px")
events.schema.content["muons"].content.rename("Muon_Py", "py")
events.schema.content["muons"].content.rename("Muon_Pz", "pz")
events.schema.content["muons"].content.rename("Muon_E", "energy")
events.schema.content["muons"].content.rename("Muon_Iso", "isolation")
events.schema.content["muons"].content.rename("Muon_Charge", "charge")
events.schema.content.rename("NPhoton", "photons")
events.schema.content["photons"].content.rename("Photon_Px", "px")
events.schema.content["photons"].content.rename("Photon_Py", "py")
events.schema.content["photons"].content.rename("Photon_Pz", "pz")
events.schema.content["photons"].content.rename("Photon_E", "energy")
events.schema.content["photons"].content.rename("Photon_Iso", "isolation")
events.schema.content.rename("NJet", "jets")
events.schema.content["jets"].content.rename("Jet_Px", "px")
events.schema.content["jets"].content.rename("Jet_Py", "py")
events.schema.content["jets"].content.rename("Jet_Pz", "pz")
events.schema.content["jets"].content.rename("Jet_E", "energy")
events.schema.content["jets"].content.rename("Jet_ID", "id")
events.schema.content["jets"].content.rename("Jet_btag", "btag")
events.regenerate()

def mass(*particles):
	energy = particles.map(lambda particle: particle.energy).sum
	px = particles.map(lambda particle: particle.px).sum
	py = particles.map(lambda particle: particle.py).sum
	pz = particles.map(lambda particle: particle.pz).sum
	return sqrt(energy**2 - px**2 - py**2 - pz**2)

def mass_energy_momentum(*particles):
	energy = particles.map(lambda particle: particle.energy).sum
	px = particles.map(lambda particle: particle.px).sum
	py = particles.map(lambda particle: particle.py).sum
	pz = particles.map(lambda particle: particle.pz).sum
	return [sqrt(energy**2 - px**2 - py**2 - pz**2), energy, px, py, pz]

def mass_from_mep(*meps):
	# mep: [mass, energy, px, py, pz]
	energy = meps.map(lambda mep: mep[1]).sum
	px = meps.map(lambda mep: mep[2]).sum
	py = meps.map(lambda mep: mep[3]).sum
	pz = meps.map(lambda mep: mep[4]).sum
	return sqrt(energy**2 - px**2 - py**2 - pz**2)
	


#------------------------------------
# mu, e -> Find Higgs
#------------------------------------
# We know that one Higgs decays into two Z bosons. 
# Each of Z bosons decays into either a pair of muons, or a pair of electrons.
# We can first get a list of all muons and electron pairs, among which some resemble Z bosons
# Then we pair up those pairs to find Higgs bosons using the given derived mass function.
def higgs_mass_pairs(muons, electrons):
	all_pairs = muons.pairs(lambda x ,y: (x ,y)) 
	el = electrons.pairs(lambda x, y: (x ,y))
	all_pairs.extend(el)
	return all_pairs.pairs(lambda x, y: mass(*(x+y)))

masses = (events.lazy
			.filter(lambda event: event.muons.size + event.electrons.size >= 4)
			.map(lambda event: higgs_mass_pairs(event.muons, event.electrons)))

# plt.hist(masses.collect.flatten, rwidth=0.5)
# plt.show()
# print(masses.collect.flatten.size)

#------------------------------------
# mu, e -> Z -> Find Higgs
#------------------------------------
def z_mep_pairs(muons, electrons):
	all_pairs = muons.pairs(mass_energy_momentum)
	el = electrons.pairs(mass_energy_momentum)
	all_pairs.extend(el)
	# return [mass, energy, momentum] of every pair in every events
	return all_pairs

def get_min_max(x, y):
	if type(x) is tuple:
		if x[1] < y:
			return (x[0], y)
		if x[0] > y:
			return (y, x[1])
		else:
			return x
	else:
		return (x,y) if x < y else (y,x)

# Take a list of data and calculate histogram range
# bin_width is the range of each interval (bucket) 
def histogram(data, bin_width=5):
	dat_min, dat_max = data.reduce(get_min_max)
	dat_min = dat_min - (dat_min % bin_width)
	hist_range = dat_max - dat_min
	# number of bins
	hist_size = int((hist_range + bin_width) / bin_width)
	# array of lower border of a bin interval
	intervals = range(hist_size).map(lambda x: dat_min + x * bin_width)
	# count of data that falls in the corresponding interval
	hist = list(np.zeros(hist_size, dtype=(int)))
	for dat in data:
		ind = int((dat - dat_min) / bin_width)
		hist[ind] = hist[ind] + 1

	peak = hist.index(max(hist)) * bin_width + dat_min
	return hist, (peak, peak+bin_width), intervals

# events --> event {mu: , e: ...}
# return [e1[pair1[],pair2[]], e2[pair1[mep],pair2[mep],pair..], e...]
# First, events where the number muons and electrons is less than four can be filtered out since Higgs will decays in to four particles (muons/electrons)
# Then muons are paired up with muons and electrons are paired up with electrons.
# To identify a particle that each pair decayed from, mass is calculated from energy and momentum. 
# (Energy and momentum are also returned as it will be used in the next step)
# So this list contains mass, energy and momentum of every muon/electron pair in every event
mue_pairs_mep = (events.lazy
					.filter(lambda event: event.muons.size + event.electrons.size >= 4) 
						# 43 events
					.map(lambda event: z_mep_pairs(event.muons, event.electrons)) 
						# 119 pairs in 43 events
					# .flatten
					.collect)

# To identify Z bosons, histogram is constructed from masses all those muon pairs and electron pairs.
# Histogram peak indicates Z bosons, whose mass range is then used to filter muon/electron pairs.
mue_pairs_masses = mue_pairs_mep.flatten.map(lambda pair: pair[0])
z_hist, (z_peak_min, z_peak_max), _ = histogram(mue_pairs_masses)

# Only muon/electron pairs that compose Z bosons are selected and one event must have more than two pairs.
# (Since Higgs bosons decay into two Z bosons)
# Masses of each Z bosons pairs are calculated to find a mass of original particle. 
# Lastly, we can identify whether those pair are decayed from Z bosons using histogram plot.

z_pairs_mep = (mue_pairs_mep
			.map(lambda event: event
				.filter(lambda pair: pair[0] >= z_peak_min and pair[0] <= z_peak_max))
			.filter(lambda event: event.size >= 2)
		)

#print(z_pairs_mep.size)

higgs_masses = (z_pairs_mep
				.map(lambda event: event
					# x = [m, e_sq, p_sq]
					.pairs(lambda x, y: mass_from_mep(x,y)))
					# .pairs(lambda x, y: sqrt((x[1]+y[1])-(x[2]+y[2]))))
				# )
				.flatten)

h_hist, (h_peak_min, h_peak_max), h_intv = histogram(higgs_masses, 10)
plt.hist(higgs_masses.collect, h_intv, rwidth=0.5)
plt.show()


# print(histogram(z_masses.take(100)))

# print (events.size)
# print (masses.collect.size)