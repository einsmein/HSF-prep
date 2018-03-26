# Solution 1: vectorized HEP

## Background

Vectorization helps accelerate data processing by processing consecutive elements in a single instruction. HEP datasets store each attribute in an array (essentially a column of a matrix of attributes), making it suitable for vector operations and exploiting cache from using contiguous memory.

We can quickly get different measures for every particle. `starts` and `stops` mark starting and ending index of particles in the same event.
```python
Muon_E = columns["Muon_E"].content
Muon_Px = columns["Muon_Px"].content
Muon_Py = columns["Muon_Py"].content
Muon_Pz = columns["Muon_Pz"].content

starts = columns["Muon_Px"].starts
stops = columns["Muon_Px"].stops
```


## Baby step

To understand the dataset and calculation better, the most straightforward solution -- *the baby step* was taken.

We need to pair up muons in each event to calculate Z mass. Vectorization index can't be on the number of muons since we won't be able to access other muons across instances of function. This leaves us with event indexing, using `starts` and `stops` to find indexes of muons in an event from the event index.

`mass` finds mass of original particles of muon pairs in each event. A `combinations` function provided by python was used to get a list of tuples of two muon indexes. Each tuple represents a pair of muons, and the indexes can be used to get information of muons i.e. their energy and momentum. The loop goes through each pair and compute mass from corresponding.

```python
def mass(index, starts, stops, Muon_E, Muon_Px, Muon_Py, Muon_Pz, Pair_M):
	index_range = range(starts[index], stops[index])
	pair_index_list = list(combinations(index_range, 2))
	Event_Pair_M = np.empty(len(pair_index_list))
	for i in range(len(pair_index_list)):
		Event_Pair_M[i] = np.sqrt((Muon_E[pair_index_list[i][0]] + Muon_E[pair_index_list[i][1]])**2
				- (Muon_Px[pair_index_list[i][0]] + Muon_Px[pair_index_list[i][1]])**2
				- (Muon_Py[pair_index_list[i][0]] + Muon_Py[pair_index_list[i][1]])**2
				- (Muon_Pz[pair_index_list[i][0]] + Muon_Pz[pair_index_list[i][1]])**2)

	Pair_M[index] = list(Event_Pair_M)


Pair_M = np.empty(len(starts), dtype=(object))
vectorize(mass, len(starts), starts, stops, Muon_E, Muon_Px, Muon_Py, Muon_Pz, Pair_M)
```
CHECK!! ******** Pair_M = Pair_M.flatten 







