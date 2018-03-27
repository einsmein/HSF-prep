from vectorized import vectorize
import uproot
import numpy as np
from itertools import combinations
import matplotlib.pyplot as plt
import time
import functional

columnar_events = uproot.open("http://scikit-hep.org/uproot/examples/HZZ.root")["events"]
columns = columnar_events.arrays(["*Muon*"])

Muon_E = columns["Muon_E"].content
Muon_Px = columns["Muon_Px"].content
Muon_Py = columns["Muon_Py"].content
Muon_Pz = columns["Muon_Pz"].content

starts = columns["Muon_Px"].starts
stops = columns["Muon_Px"].stops



# ======================================================================
# Examples
# ======================================================================
def totalp(index, Muon_Px, Muon_Py, Muon_Pz, Muon_P):
	px2 = Muon_Px[index]**2
	py2 = Muon_Py[index]**2
	pz2 = Muon_Pz[index]**2
	Muon_P[index] = np.sqrt(px2 + py2 + pz2)

Muon_P = np.empty(len(Muon_Px))
# vectorize(totalp, len(Muon_Px), Muon_Px, Muon_Py, Muon_Pz, Muon_P)


def maxp(index, starts, stops, Muon_P, highest_by_event):
	highest = float("nan")
	for i in range(starts[index], stops[index]):
		if np.isnan(highest) or Muon_P[i] > highest:
			highest = Muon_P[i]
	highest_by_event[index] = highest

highest_by_event = np.empty(len(starts))
# vectorize(maxp, len(starts), starts, stops, Muon_P, highest_by_event)


# ======================================================================
# Baby steps
# ======================================================================
def baby_step(index, Pair_M, starts, stops, Muon_E, Muon_Px, Muon_Py, Muon_Pz):
	# Pair_E = list(combinations(Muon_E[starts[index]:stops[index]], 2))
	# Pair_Px = list(combinations(Muon_Px[starts[index]:stops[index]], 2))
	# Pair_Py = list(combinations(Muon_Py[starts[index]:stops[index]], 2))
	# Pair_Pz = list(combinations(Muon_Pz[starts[index]:stops[index]], 2))

	# Event_Pair_M = np.empty(len(Pair_E))
	# for i in range(len(Pair_E)):
	# 	Event_Pair_M[i] = np.sqrt(Pair_E[i].sum**2 - Pair_Px[i].sum**2 - Pair_Py[i].sum**2 - Pair_Pz[i].sum**2)
	
	index_range = range(starts[index], stops[index])
	pair_index_list = list(combinations(index_range, 2))
	Event_Pair_M = np.empty(len(pair_index_list))
	for i in range(len(pair_index_list)):
		Event_Pair_M[i] = np.sqrt((Muon_E[pair_index_list[i][0]] + Muon_E[pair_index_list[i][1]])**2
							- (Muon_Px[pair_index_list[i][0]] + Muon_Px[pair_index_list[i][1]])**2
							- (Muon_Py[pair_index_list[i][0]] + Muon_Py[pair_index_list[i][1]])**2
							- (Muon_Pz[pair_index_list[i][0]] + Muon_Pz[pair_index_list[i][1]])**2)

	Pair_M[index] = Event_Pair_M


# Z_M = np.empty(len(starts))
# zmass = list(starts).vmap(best_Z, Z_M, stops, Muon_E, Muon_Px, Muon_Py, Muon_Pz)

Pair_M = np.empty(len(starts), dtype=(object))
zmass = list(starts).vmap(baby_step, Pair_M, stops, Muon_E, Muon_Px, Muon_Py, Muon_Pz)
# vectorize(baby_step, len(starts), starts, stops, Muon_E, Muon_Px, Muon_Py, Muon_Pz, Pair_M)
zmass = np.concatenate(zmass)


# ======================================================================
# Pair listing
# ======================================================================
def pair_listing(index, starts, stops, Muon_E, Muon_Px, Muon_Py, Muon_Pz, Pair_M):
	N = stops[index]-starts[index]
	NPair = int(N*(N-1)/2)
	# Pair_E = np.empty(NPair)
	# Pair_Px = np.empty(NPair)
	# Pair_Py = np.empty(NPair)
	# Pair_Pz = np.empty(NPair)
	Event_Pair_M = np.empty(NPair)

	i = 0
	for j in reversed(range(1, N)):
		# Pair_E[i:i+j] = (Muon_E[stops[index]-j-1] * j + Muon_E[stops[index]-j:stops[index]])**2
		# Pair_Px[i:i+j] = (Muon_Px[stops[index]-j-1] * j + Muon_Px[stops[index]-j:stops[index]])**2
		# Pair_Py[i:i+j] = (Muon_Py[stops[index]-j-1] * j + Muon_Py[stops[index]-j:stops[index]])**2
		# Pair_Pz[i:i+j] = (Muon_Pz[stops[index]-j-1] * j + Muon_Pz[stops[index]-j:stops[index]])**2
		Event_Pair_M[i:i+j] = (Muon_E[stops[index]-j-1] * j + Muon_E[stops[index]-j:stops[index]])**2
		Event_Pair_M[i:i+j] = Event_Pair_M[i:i+j] - (Muon_Px[stops[index]-j-1] * j + Muon_Px[stops[index]-j:stops[index]])**2
		Event_Pair_M[i:i+j] = Event_Pair_M[i:i+j] - (Muon_Py[stops[index]-j-1] * j + Muon_Py[stops[index]-j:stops[index]])**2
		Event_Pair_M[i:i+j] = Event_Pair_M[i:i+j] - (Muon_Pz[stops[index]-j-1] * j + Muon_Pz[stops[index]-j:stops[index]])**2
		i = i + j

	# Pair_M[index] = np.sqrt(Pair_E - Pair_Px - Pair_Py - Pair_Pz)
	Pair_M[index] = np.sqrt(Event_Pair_M)

# ======================================================================
# divided vectorized steps
# ======================================================================

def get_pair_index(index, starts, stops, pair_index_per_event):
	index_range = range(starts[index], stops[index])
	pair_index_per_event[index] = list(combinations(index_range, 2))

def get_pair_mass(index, pair_index, Muon_E, Muon_Px, Muon_Py, Muon_Pz, Pair_M):
	Pair_M[index] = np.sqrt((Muon_E[pair_index[index][0]] + Muon_E[pair_index[index][1]])**2
			- (Muon_Px[pair_index[index][0]] + Muon_Px[pair_index[index][1]])**2
			- (Muon_Py[pair_index[index][0]] + Muon_Py[pair_index[index][1]])**2
			- (Muon_Pz[pair_index[index][0]] + Muon_Pz[pair_index[index][1]])**2)


# ======================================================================
# Best Z candidate
# ======================================================================
def best_Z(index, Z_M, starts, stops, Muon_E, Muon_Px, Muon_Py, Muon_Pz):
	index_range = range(starts[index], stops[index])
	pair_index_list = list(combinations(index_range, 2))
	Event_Pair_M = np.zeros(len(pair_index_list))
	mass_Z = -1
	for i in range(len(pair_index_list)):
		Event_Pair_M[i] = (
		# mass = (
					np.sqrt((Muon_E[pair_index_list[i][0]] + Muon_E[pair_index_list[i][1]])**2
						- (Muon_Px[pair_index_list[i][0]] + Muon_Px[pair_index_list[i][1]])**2
						- (Muon_Py[pair_index_list[i][0]] + Muon_Py[pair_index_list[i][1]])**2
						- (Muon_Pz[pair_index_list[i][0]] + Muon_Pz[pair_index_list[i][1]])**2)
				)
		# if abs(mass-91) < abs(mass-mass_Z):
		# 	mass_Z = mass

	if len(pair_index_list) > 0:
		mass_Z = Event_Pair_M[np.argmin(np.abs(Event_Pair_M-91))] # mass_Z
	Z_M[index] = mass_Z

# Z_M = np.empty(len(starts))
# zmass = list(starts).vmap(best_Z, Z_M, stops, Muon_E, Muon_Px, Muon_Py, Muon_Pz)


# ======================================================================
# Interfaces
# ======================================================================

def Z_mass_from_baby_step(starts, stops, Muon_E, Muon_Px, Muon_Py, Muon_Pz):
	Pair_M = np.empty(len(starts), dtype=(object))
	vectorize(baby_step, len(starts), Pair_M, starts, stops, Muon_E, Muon_Px, Muon_Py, Muon_Pz)
	return np.concatenate(Pair_M)

def Z_mass_from_pair_listing(starts, stops, Muon_E, Muon_Px, Muon_Py, Muon_Pz):
	Pair_M = np.empty(len(starts), dtype=(object))
	vectorize(pair_listing, len(starts), starts, stops, Muon_E, Muon_Px, Muon_Py, Muon_Pz, Pair_M)
	return np.concatenate(Pair_M)

def Z_mass_from_divided_steps(starts, stops, Muon_E, Muon_Px, Muon_Py, Muon_Pz):
	pair_index_per_event = np.empty(len(starts), dtype=(object))
	vectorize(get_pair_index, len(starts), starts, stops, pair_index_per_event)
	pair_index = list(pair_index_per_event).flatten

	Pair_M = np.empty(len(pair_index))
	vectorize(get_pair_mass, len(pair_index), pair_index, Muon_E, Muon_Px, Muon_Py, Muon_Pz, Pair_M)
	return Pair_M

def best_Z_candidate(starts, stops, Muon_E, Muon_Px, Muon_Py, Muon_Pz):
	Z_M = np.empty(len(starts))
	vectorize(best_Z, len(starts), Z_M, starts, stops, Muon_E, Muon_Px, Muon_Py, Muon_Pz)
	return Z_M[Z_M > np.array(0)]

	

# ======================================================================
# Calling interfaces
# ======================================================================
start = time.time()
# zmass = Z_mass_from_baby_step(starts, stops, Muon_E, Muon_Px, Muon_Py, Muon_Pz)
# zmass = Z_mass_from_pair_listing(starts, stops, Muon_E, Muon_Px, Muon_Py, Muon_Pz)
# zmass = Z_mass_from_divided_steps(starts, stops, Muon_E, Muon_Px, Muon_Py, Muon_Pz)
# zmass = best_Z_candidate(starts, stops, Muon_E, Muon_Px, Muon_Py, Muon_Pz)
stop = time.time()
print(stop-start)


# ======================================================================
# Display result
# ======================================================================
# print(Pair_M)
binwidth = 5
print(zmass.size)
plt.hist(zmass, bins=range(int(zmass.min()), int(zmass.max()) + binwidth, binwidth))
plt.show()


# Pair_E = list(combinations(Muon_E[starts[0]:stops[0]], 2))
# Pair_Px = list(combinations(Muon_Px[starts[0]:stops[0]], 2))
# Pair_Py = list(combinations(Muon_Py[starts[0]:stops[0]], 2))
# Pair_Pz = list(combinations(Muon_Pz[starts[0]:stops[0]], 2))
# print(Pair_E[0].sum, Pair_Px[0].sum, Pair_Py[0].sum, Pair_Pz[0].sum)
# print(np.sqrt(Pair_E[0].sum**2 - Pair_Px[0].sum**2 - Pair_Py[0].sum**2 - Pair_Pz[0].sum**2))