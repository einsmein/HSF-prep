from vectorized import vectorize
import uproot
import numpy as np
from itertools import combinations
import matplotlib.pyplot as plt
import time


columnar_events = uproot.open("http://scikit-hep.org/uproot/examples/HZZ.root")["events"]
columns = columnar_events.arrays(["*Muon*"])

Muon_E = columns["Muon_E"].content
Muon_Px = columns["Muon_Px"].content
Muon_Py = columns["Muon_Py"].content
Muon_Pz = columns["Muon_Pz"].content

starts = columns["Muon_Px"].starts
stops = columns["Muon_Px"].stops

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
# divide steps
# ======================================================================

def pair(index, starts, stops, Muon_E, Muon_Px, Muon_Py, Muon_Pz, Pair_E, Pair_Px, Pair_Py, Pair_Pz):
	Pair_E[index] = list(combinations(Muon_E[starts[index]:stops[index]], 2))
	Pair_Px[index] = list(combinations(Muon_Px[starts[index]:stops[index]], 2))
	Pair_Py[index] = list(combinations(Muon_Py[starts[index]:stops[index]], 2))
	Pair_Pz[index] = list(combinations(Muon_Pz[starts[index]:stops[index]], 2))

def pair_mass(index, Pair_E, Pair_Px, Pair_Py, Pair_Pz, Pair_M):
	Pair_M[index] = np.sqrt(Pair_E[index].sum**2 - Pair_Px[index].sum**2 - Pair_Py[index].sum**2 - Pair_Pz[index].sum**2)

# start = time.time()
# Pair_E = np.empty(len(starts), dtype=(object))
# Pair_Px = np.empty(len(starts), dtype=(object))
# Pair_Py = np.empty(len(starts), dtype=(object))
# Pair_Pz = np.empty(len(starts), dtype=(object))
# vectorize(pair, len(starts), starts, stops, Muon_E, Muon_Px, Muon_Py, Muon_Pz, Pair_E, Pair_Px, Pair_Py, Pair_Pz)

# Pair_E = list(Pair_E).flatten
# Pair_Px = list(Pair_Px).flatten
# Pair_Py = list(Pair_Py).flatten
# Pair_Pz = list(Pair_Pz).flatten
# Pair_M = np.empty(len(Pair_E))
# vectorize(pair_mass, len(Pair_E), Pair_E, Pair_Px, Pair_Py, Pair_Pz, Pair_M)

# stop = time.time()
# print(stop-start)

# ======================================================================
# Baby steps
# ======================================================================
def baby_step(index, starts, stops, Muon_E, Muon_Px, Muon_Py, Muon_Pz, Pair_M):
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
# Interfaces
# ======================================================================
def mass_from_baby_step():
	Pair_M = np.empty(len(starts), dtype=(object))
	vectorize(baby_step, len(starts), starts, stops, Muon_E, Muon_Px, Muon_Py, Muon_Pz, Pair_M)
	return np.concatenate(Pair_M)

def mass_from_pair_listing():
	Pair_M = np.empty(len(starts), dtype=(object))
	vectorize(pair_listing, len(starts), starts, stops, Muon_E, Muon_Px, Muon_Py, Muon_Pz, Pair_M)
	return np.concatenate(Pair_M)

# ======================================================================
# Calling interfaces
# ======================================================================
# start = time.time()
# Pair_M = mass_from_baby_step()
Pair_M = mass_from_pair_listing()
# stop = time.time()
# print(stop-start)
# print(Pair_M.size)

# print(Pair_M)

# ======================================================================
# Display result
# ======================================================================
# print(Pair_M)
binwidth = 5
print(Pair_M)
plt.hist(Pair_M, bins=range(int(Pair_M.min()), int(Pair_M.max()) + binwidth, binwidth))
plt.show()


# Pair_E = list(combinations(Muon_E[starts[0]:stops[0]], 2))
# Pair_Px = list(combinations(Muon_Px[starts[0]:stops[0]], 2))
# Pair_Py = list(combinations(Muon_Py[starts[0]:stops[0]], 2))
# Pair_Pz = list(combinations(Muon_Pz[starts[0]:stops[0]], 2))
# print(Pair_E[0].sum, Pair_Px[0].sum, Pair_Py[0].sum, Pair_Pz[0].sum)
# print(np.sqrt(Pair_E[0].sum**2 - Pair_Px[0].sum**2 - Pair_Py[0].sum**2 - Pair_Pz[0].sum**2))