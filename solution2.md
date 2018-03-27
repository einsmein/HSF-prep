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


## The Baby step

To understand the dataset and calculation better, the most straightforward solution -- *the baby step* was taken.

We need to pair up muons in each event to calculate Z mass. Vectorization index can't be on the number of muons since we won't be able to access other muons across instances of functions. This leaves us with event indexing, using `starts` and `stops` to find indexes of muons in an event from the event index.

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

	Pair_M[index] = Event_Pair_M

```

```python
>>> Pair_M = np.empty(len(starts), dtype=(object))
>>> vectorize(mass, len(starts), starts, stops, Muon_E, Muon_Px, Muon_Py, Muon_Pz, Pair_M)
leading step 0 (100.0% at leading): 
    index_range = range(starts[index], stops[index])
    ...advancing 1

leading step 1 (100.0% at leading): 
    pair_index_list = list(combinations(index_range, 2))
    ...advancing 2

leading step 2 (100.0% at leading): 
    Event_Pair_M = np.empty(len(pair_index_list))
    ...advancing 3

leading step 3 (100.0% at leading): 
    for i in range(len(pair_index_list)):
        Event_Pair_M[i] = np.sqrt((((((Muon_E[pair_index_list[i][0]] + Muon_E[pair_index_list[i][1]]) ** 2) - ((Muon_Px[pair_index_list[i][0]] + Muon_Px[pair_index_list[i][1]]) ** 2)) - ((Muon_Py[pair_index_list[i][0]] + Muon_Py[pair_index_list[i][1]]) ** 2)) - ((Muon_Pz[pair_index_list[i][0]] + Muon_Pz[pair_index_list[i][1]]) ** 2)))
    ...advancing 4

leading step 5 (41.64% at leading): 
    Pair_M[index] = Event_Pair_M
    ...catching up 5 (41.64% at leading)
    ...catching up 6 (98.27% at leading)
    ...catching up 7 (98.27% at leading)
    ...catching up 8 (99.67% at leading)
    ...catching up 9 (99.67% at leading)
    ...catching up 10 (99.67% at leading)
    ...advancing 11

```

Taking a [closer look](https://docs.python.org/2/library/itertools.html#itertools.combinations) at `combination` function, the execution time is a second degree polynomial. So this approach contains two loops, one iterates for N square times and one for NPair times (where N = # muons, NPair = # muon pairs).

## Pair Listing

Since we can't find a way to index on a different scale (yet), we can try to speed up this computation by reduce the number of loops and their length in a function. To reduce it to a linear time, we can create combination by going through the list only once (hence taking linear time). The idea is that the first item is paired with the other n-1 items, the second item is paired with the other n-2, and so on. From that we can create two arrays whose length is the number of pairs. For example, if there are four elements, we can create

![](https://snag.gy/0yGSvB.jpg)
![](https://snag.gy/uyMpjN.jpg)

Each pair is denoted by array index. Elements at the same index in two arrays is a pair. Using numpy vectorized array operations, we can perform mass calculation while retrieving the pair arrays.

```python
def pair(index, starts, stops, Muon_E, Muon_Px, Muon_Py, Muon_Pz, Pair_M):
	N = stops[index]-starts[index]
	NPair = int(N*(N-1)/2)
	Event_Pair_M = np.empty(NPair)

	i = 0
	for j in reversed(range(1, N)):
		Event_Pair_M[i:i+j] = (Muon_E[stops[index]-j-1] * j + Muon_E[stops[index]-j:stops[index]])**2
		Event_Pair_M[i:i+j] = Event_Pair_M[i:i+j] - (Muon_Px[stops[index]-j-1] * j + Muon_Px[stops[index]-j:stops[index]])**2
		Event_Pair_M[i:i+j] = Event_Pair_M[i:i+j] - (Muon_Py[stops[index]-j-1] * j + Muon_Py[stops[index]-j:stops[index]])**2
		Event_Pair_M[i:i+j] = Event_Pair_M[i:i+j] - (Muon_Pz[stops[index]-j-1] * j + Muon_Pz[stops[index]-j:stops[index]])**2
		
	Pair_M[index] = np.sqrt(Event_Pair_M)
```


```
>>> Pair_M = np.empty(len(starts), dtype=(object))
>>> vectorize(pair_listing, len(starts), starts, stops, Muon_E, Muon_Px, Muon_Py, Muon_Pz, Pair_M)
leading step 0 (100.0% at leading): 
    N = (stops[index] - starts[index])
    ...advancing 1

leading step 1 (100.0% at leading): 
    NPair = int(((N * (N - 1)) / 2))
    ...advancing 2

leading step 2 (100.0% at leading): 
    Event_Pair_M = np.empty(NPair)
    ...advancing 3

leading step 3 (100.0% at leading): 
    i = 0
    ...advancing 4

leading step 4 (100.0% at leading): 
    for j in reversed(range(1, N)):
        Event_Pair_M[i:(i + j)] = (((Muon_E[((stops[index] - j) - 1)] * j) + Muon_E[(stops[index] - j):stops[index]]) ** 2)
        Event_Pair_M[i:(i + j)] = (Event_Pair_M[i:(i + j)] - (((Muon_Px[((stops[index] - j) - 1)] * j) + Muon_Px[(stops[index] - j):stops[index]]) ** 2))
        Event_Pair_M[i:(i + j)] = (Event_Pair_M[i:(i + j)] - (((Muon_Py[((stops[index] - j) - 1)] * j) + Muon_Py[(stops[index] - j):stops[index]]) ** 2))
        Event_Pair_M[i:(i + j)] = (Event_Pair_M[i:(i + j)] - (((Muon_Pz[((stops[index] - j) - 1)] * j) + Muon_Pz[(stops[index] - j):stops[index]]) ** 2))
        i = (i + j)
    ...advancing 5

leading step 10 (41.64% at leading): 
    Pair_M[index] = np.sqrt(Event_Pair_M)
    ...catching up 6 (41.64% at leading)
    ...catching up 7 (41.64% at leading)
    ...catching up 8 (41.64% at leading)
    ...catching up 9 (41.64% at leading)
    ...catching up 10 (41.64% at leading)
    ...catching up 11 (98.27% at leading)
    ...catching up 12 (98.27% at leading)
    ...catching up 13 (98.27% at leading)
    ...catching up 14 (98.27% at leading)
    ...catching up 15 (98.27% at leading)
    ...catching up 16 (99.67% at leading)
    ...catching up 17 (99.67% at leading)
    ...catching up 18 (99.67% at leading)
    ...catching up 19 (99.67% at leading)
    ...catching up 20 (99.67% at leading)
    ...advancing 21

```

This approach does not perform very well. From rough execution time measurement, it takes as long as the previous approach. This could be due to more operation loads in the loop even though there is only one loop with N iteration.


## Baby Step Revisit: Divided Vectorized Functions
Since using index pairs as in the baby step seemed like a smart idea, we use that and modify the loop instead. One way to get rid of for loop altogether is to use two vectorized function. First indexes of muon pairs are obtained for every event. Flattening that list, we get all muon pairs indexes across all events.

```python
def get_pair_index(index, starts, stops, pair_index_per_event):
	index_range = range(starts[index], stops[index])
	pair_index_per_event[index] = list(combinations(index_range, 2))

pair_index_per_event = np.empty(len(starts), dtype=(object))
vectorize(get_pair_index, len(starts), starts, stops, pair_index_per_event)
pair_index = list(pair_index_per_event).flatten
```

Now we can just index on the number of muon pairs, and compute Z mass from energy and momentum using the indexes from before

```python
def get_pair_mass(index, pair_index, Muon_E, Muon_Px, Muon_Py, Muon_Pz, Pair_M):
	Pair_M[index] = np.sqrt((Muon_E[pair_index[index][0]] + Muon_E[pair_index[index][1]])**2
			- (Muon_Px[pair_index[index][0]] + Muon_Px[pair_index[index][1]])**2
			- (Muon_Py[pair_index[index][0]] + Muon_Py[pair_index[index][1]])**2
			- (Muon_Pz[pair_index[index][0]] + Muon_Pz[pair_index[index][1]])**2)


Pair_M = np.empty(len(pair_index))
vectorize(get_pair_mass, len(pair_index), pair_index, Muon_E, Muon_Px, Muon_Py, Muon_Pz, Pair_M)
```

This method was the fastest with least vectorized steps among the three.

```python
leading step 0 (100.0% at leading): 
    index_range = range(starts[index], stops[index])
    ...advancing 1

leading step 1 (100.0% at leading): 
    pair_index_per_event[index] = list(combinations(index_range, 2))
    ...advancing 2

leading step 0 (100.0% at leading): 
    Pair_M[index] = np.sqrt((((((Muon_E[pair_index[index][0]] + Muon_E[pair_index[index][1]]) ** 2) - ((Muon_Px[pair_index[index][0]] + Muon_Px[pair_index[index][1]]) ** 2)) - ((Muon_Py[pair_index[index][0]] + Muon_Py[pair_index[index][1]]) ** 2)) - ((Muon_Pz[pair_index[index][0]] + Muon_Pz[pair_index[index][1]]) ** 2)))
    ...advancing 1
```

## Best Z candidate

To find the best Z candidate, mass is computed for each pair using the same method as in The Baby Step (since it was much simpler than the other). 

```python
def best_Z(index, starts, stops, Muon_E, Muon_Px, Muon_Py, Muon_Pz, Z_M):
	index_range = range(starts[index], stops[index])
	pair_index_list = list(combinations(index_range, 2))
	Event_Pair_M = np.zeros(len(pair_index_list))
	mass_Z = -1
	for i in range(len(pair_index_list)):
		Event_Pair_M[i] = np.sqrt((Muon_E[pair_index_list[i][0]] + Muon_E[pair_index_list[i][1]])**2
			- (Muon_Px[pair_index_list[i][0]] + Muon_Px[pair_index_list[i][1]])**2
			- (Muon_Py[pair_index_list[i][0]] + Muon_Py[pair_index_list[i][1]])**2
			- (Muon_Pz[pair_index_list[i][0]] + Muon_Pz[pair_index_list[i][1]])**2)

	if len(pair_index_list) > 0:
		mass_Z = Event_Pair_M[np.argmin(np.abs(Event_Pair_M-91))]
	Z_M[index] = mass_Z
```

```python
>>> Z_M = np.empty(len(starts))
>>> vectorize(best_Z, len(starts), starts, stops, Muon_E, Muon_Px, Muon_Py, Muon_Pz, Z_M)
leading step 0 (100.0% at leading): 
    index_range = range(starts[index], stops[index])
    ...advancing 1

leading step 1 (100.0% at leading): 
    pair_index_list = list(combinations(index_range, 2))
    ...advancing 2

leading step 2 (100.0% at leading): 
    Event_Pair_M = np.zeros(len(pair_index_list))
    ...advancing 3

leading step 3 (100.0% at leading): 
    mass_Z = -1
    ...advancing 4

leading step 4 (100.0% at leading): 
    for i in range(len(pair_index_list)):
        Event_Pair_M[i] = np.sqrt((((((Muon_E[pair_index_list[i][0]] + Muon_E[pair_index_list[i][1]]) ** 2) - ((Muon_Px[pair_index_list[i][0]] + Muon_Px[pair_index_list[i][1]]) ** 2)) - ((Muon_Py[pair_index_list[i][0]] + Muon_Py[pair_index_list[i][1]]) ** 2)) - ((Muon_Pz[pair_index_list[i][0]] + Muon_Pz[pair_index_list[i][1]]) ** 2)))
    ...advancing 5

leading step 6 (41.64% at leading): 
    if (len(pair_index_list) > 0):
        mass_Z = Event_Pair_M[np.argmin(np.abs((Event_Pair_M - 91)))]
    ...catching up 6 (41.64% at leading)
    ...catching up 7 (98.27% at leading)
    ...catching up 8 (98.27% at leading)
    ...catching up 9 (99.67% at leading)
    ...catching up 10 (99.67% at leading)
    ...catching up 11 (99.67% at leading)
    ...advancing 12

leading step 8 (41.64% at leading): 
    Z_M[index] = mass_Z
    ...catching up 13 (41.64% at leading)
    ...advancing 14

```

An alternative (shown below) is to keep track of mass closest to 91 GeV while going through the loop. With this method, execution time stayed somewhat the same but it took 24 vectorized step instead.

```python
	for i in range(len(pair_index_list)):
		mass = ...

		if abs(mass-91) < abs(mass-mass_Z):
			mass_Z = mass
	Z_M[index] = mass_Z
```



## Interfaces

Finally all the functions shown in this solution are wrapped into functional interfaces which takes muon attribute arrays from HEP dataset as input.

```python
def Z_mass_from_baby_step(starts, stops, Muon_E, Muon_Px, Muon_Py, Muon_Pz):
	Pair_M = np.empty(len(starts), dtype=(object))
	vectorize(baby_step, len(starts), starts, stops, Muon_E, Muon_Px, Muon_Py, Muon_Pz, Pair_M)
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
	vectorize(best_Z, len(starts), starts, stops, Muon_E, Muon_Px, Muon_Py, Muon_Pz, Z_M)
	return Z_M[Z_M > np.array(0)]
```

