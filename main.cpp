

#include <boost/numeric/ublas/io.hpp>



using namespace std;




namespace ublas = boost::numeric::ublas;

/*********************************************************************************/
/*                                HELPER FUNCTIONS                               */
/*********************************************************************************/

// Make Boost::Matrix from given Sizes m,n with Values v
template <typename T, typename F = ublas::row_major>
ublas::matrix<T, F> makeMatrix(size_t m, size_t n, vector<T> & v)
{
	ublas::unbounded_array<T> storage(m*n);
	copy(v.begin(), v.end(), storage.begin());
	return ublas::matrix<T>(m, n, storage);
}

// Print Vector Elements inline
template<typename T>
void printVector(const T& t)
{
	cout << "Vector: ";
	for (size_t i = 0; i != t.size(); ++i)
	{
		if (i == 0)
			cout << "( " << t[i] << " ,";
		else if (i == t.size() - 1)
			cout << " " << t[i] << " )";
		else
			cout << " " << t[i] << " ,";
	}
	cout << endl;
}

// Print Matrix Elements
template<typename T>
void printMatrix(const ublas::matrix<T> &m)
{
	cout << "Matrix: " << endl;
	for (size_t i = 0; i != m.size1(); ++i)
	{
		for (size_t j = 0; j != m.size2(); ++j)
		{
			if (j == 0)
				cout << "(" << m(i, j) << " , \t\t";
			else if(j == m.size2()-1)
				cout << " " << m(i, j) << ")";
			else
				cout << " " << m(i, j) << ", \t\t";
		}
		cout << endl;
	}
}

// Select Random index from Vector
template<typename T>
int select_randomly(const vector<T> &vec)
{
	if (vec.size() > 1)
	{
		random_device random_device;
		mt19937 engine{ random_device() };
		size_t upper = vec.size() - 1;
		uniform_int_distribution<int> dist(0, upper);

		return dist(engine);
	}

	return 0;
}

// Get Uniform Randomly Double or Integer Value
double get_random_double(const double lower, const double upper);
int get_random_int(const int lower, const int upper);

void printFinalSolution(
	const vector<bool> &yi,
	const double &fx,
	const unsigned int &kMax,
	const unsigned int &tMax,
	const unsigned int &shaking_mode,
	const unsigned int &local_search_mode,
	const unsigned int &update_xij_mode,
	const unsigned int &init_mode,
	const string &directory,
	chrono::steady_clock::time_point &bvns_start,
	chrono::steady_clock::time_point &bvns_end); // Print Solution Functions






/*********************************************************************************/
/*                        BVNS CLASS, Inherit from VNS                          */
/*********************************************************************************/

class BVNS : public VNS
{
public:

	/*********************************************************************************/
	/*                                 CONSTRUCTOR                                   */
	/*********************************************************************************/

	BVNS(const string &directory, 
		const unsigned int &kMin,
		const unsigned int &kMax,
		const unsigned int &tMax,
		const unsigned int &shaking_mode,
		const unsigned int &local_Search_mode,
		const unsigned int &update_xij_mode,
		const unsigned int &initial_mode,
		const unsigned int &time); // Init BVNS Heuristic

	/*********************************************************************************/
	/*                FIRST OR BEST IMPROVMENT LOCAL SEARCH                          */
	/*********************************************************************************/

	vector<bool> localSearch(
		const vector<bool> &perturbed_solution,
		const vector<vector<bool>> &neighborhood_k,
		const unsigned int &k,
		double &fx,
		const unsigned int &local_search_mode); // Local Search Procedure

	vector<bool> localSearchFirstImprovment(
		const vector<bool> &perturbed_solution,
		const vector<vector<bool>> &neighborhood_k,
		const unsigned int &k,
		double &fx); // Search First Improvment in N_k(S)

	vector<bool> localSearchBestImprovment(
		const vector<bool> &perturbed_solution,
		const vector<vector<bool>> &neighborhood_k,
		const unsigned int &k,
		double &fx); // Search Best Improvment in N_k(S)

/*********************************************************************************/
/*                              MEMBER VARIABLES                                */
/*********************************************************************************/

private:
	vector<bool> m_local_solution; // Local Minimum Solution
};






namespace ublas = boost::numeric::ublas;

namespace t_simplex {

	/* DECLARATION OF DATA TYPES */
	enum TsError { TsErrBadInput };

	//TsSignature is used for inputting the source and sink signatures

	class TsSignature
	{
	public:
		unsigned int n;									// Number of features in the signature 
		int *features;							// Pointer to the features vector 
		double *weights;						// Pointer to the weights of the features 
		TsSignature(unsigned int nin, int *fin, double * win) :n(nin), features(fin), weights(win) {};
	};

	//TsFlow is used for outputting the final flow table
	typedef struct TsFlow
	{
		int from;								// Feature number in signature 1 
		int to;									// Feature number in signature 2 
		double amount;							// Amount of flow from signature1.features[from] to signature2.features[to]
	} TsFlow;

	// TsBasic is used for 2D lists, allowing for easy navigation of the basic variables 
	typedef struct TsBasic
	{
		int i, j;
		double val;
		TsBasic *nextCus, *prevCus;				//next/previous node in the column
		TsBasic *nextFac, *prevFac;				//next/previous node in the row
	} TsBasic;

	// TsStone is used for _BFS
	typedef struct TsStone
	{
		struct TsStone *prev;
		struct TsBasic *node;
	} TsStone;

	// TsVogPen is used for 1D lists in _initVogel
	typedef struct TsVogPen
	{
		int i;
		struct TsVogPen *next, *prev;
		int one, two;
		double oneCost, twoCost;
	} TsVogPen;

	// Helper function for _initVogel
	inline void addPenalty(
		TsVogPen * pitr,
		double &cost,
		int &i)
	{
		if (pitr != NULL)
		{
			if (cost < pitr->oneCost)
			{
				pitr->twoCost = pitr->oneCost;
				pitr->two = pitr->one;
				pitr->oneCost = cost;
				pitr->one = i;
			}
			else if (cost < pitr->twoCost)
			{
				pitr->twoCost = cost;
				pitr->two = i;
			}
		}
	}

	/* DECLARATIONS */
	double transportSimplex(
		const unsigned &update_xij_mode,
		ublas::matrix<double> &custom_cij,
		TsFlow *flowTable,
		unsigned int *flowSize,
		TsSignature *facility,
		TsSignature *customer,
		const double &sum_dj,
		const double &sum_bi,
		const vector <double > &dj);

	double _pivot(
		TsBasic * basics,
		TsBasic ** facBasics,
		TsBasic ** cusBasics,
		bool ** isBasic,
		int &m,
		int &n,
		const vector <double > &dj);

	TsStone * _BFS(
		TsStone * stoneTree,
		TsBasic ** facBasics,
		TsBasic ** cusBasics,
		bool complete = false);

	void _initVogel(
		double *S,
		double *D,
		TsBasic * basicsEnd,
		TsBasic ** facBasics,
		TsBasic ** cusBasics,
		bool ** isBasic,
		int m,
		int n);

	void _initNW(
		double *bi,
		double *dj,
		TsBasic * basicsEnd,
		TsBasic ** facBasics,
		TsBasic ** cusBasics,
		bool ** isBasic,
		int m,
		int n);

	void _initLCM(
		double *bi,
		double *dj,
		TsBasic * basicsEnd,
		TsBasic ** facBasics,
		TsBasic ** cusBasics,
		bool ** isBasic,
		int m,
		int n,
		ublas::matrix<unsigned int> &cij_column_sorted,
		int *facilities);
}





namespace ublas = boost::numeric::ublas;

/*********************************************************************************/
/*                                HELPER FUNCTIONS                               */
/*********************************************************************************/

// Get Uniform Randomly Double Value
double get_random_double(const double lower, const double upper)
{
	random_device random_device;
	mt19937 engine{ random_device() };
	uniform_real_distribution<double> dist(lower, upper);

	return dist(engine);
}

// Get Uniform Randomly Integer Value
int get_random_int(const int lower, const int upper)
{
	random_device random_device;
	mt19937 engine{ random_device() };
	uniform_int_distribution<int> dist(lower, upper);

	return dist(engine);
}

// Prints Result of y_i and x_ij
void printFinalSolution(
	const vector<bool> &yi,
	const double &fx,
	const unsigned int &kMax,
	const unsigned int &tMax,
	const unsigned int &shaking_mode,
	const unsigned int &local_search_mode,
	const unsigned int &update_xij_mode,
	const unsigned int &init_mode,
	const string &directory,
	chrono::steady_clock::time_point &bvns_start,
	chrono::steady_clock::time_point &bvns_end)
{
	//                         (0=F, 1=B)      (0=G, 1=R)
	// kMax, kMin, tM,   shake,   local,   xij,  init,     
	string solution =
			to_string(kMax) + "_"
		+	to_string(tMax) + "_"
		+	to_string(shaking_mode);

	string full_directory = directory.substr(0, directory.find("/Testdaten/")) + "/Ergebnisse/"
		+ directory.substr(directory.find_last_of("/") + 1) + "/";
	cout << endl << "Soltuion: " << solution << "  f(x) = " + to_string(fx) << endl;

	//_mkdir(full_directory.c_str());

	ofstream ofile(full_directory + solution + ".txt", ofstream::out);

	ofile << "BEGIN PARAMS" << endl;
	ofile << "kMax: " + to_string(kMax) << endl;
	ofile << "tMax: " + to_string(tMax) << endl;

	if(shaking_mode == 0)
		ofile << "Shaking: " << "shakingKOperations" << endl;
	else if (shaking_mode == 1)
		ofile << "Shaking: " << "shakingKMaxOperations" << endl;
	else if (shaking_mode == 2)
		ofile << "Shaking: " << "shakingAssignments" << endl;
	else
		ofile << "Shaking: " << "shakingCosts" << endl;

	if(local_search_mode == 0)
		ofile << "LocalSearch: " << "First Improvment" << endl;
	else
		ofile << "LocalSearch: " << "Best Improvment" << endl;

	ofile << "Update Xij: " << "Vogel Approx." << endl;

	if (init_mode == 1)
		ofile << "Init: " << "RVNS" << endl;
	else
		ofile << "Init: " << "Random" << endl;

	ofile << "Soltuion: f(x) = " + to_string(fx) << endl;
	ofile << "Overall Time: " << chrono::duration_cast<chrono::microseconds>(bvns_end - bvns_start).count() << "s" << endl;
	ofile << "END PARAMS" << endl << endl;
	
	for (size_t i = 0; i != yi.size(); ++i)
	{
		if (yi[i] == 1)
		{
			ofile << "y[" << i << "] = 1" << endl;
		}
	}

	ofile.close();
}








namespace ublas = boost::numeric::ublas;

/*********************************************************************************/
/*                  CONSTRUCTOR AND MEMBER FUNCTIONS OF BVNS                     */
/*********************************************************************************/

// Init BVNS Heuristic
BVNS::BVNS(const string &directory, 
	const unsigned int &kMin,
	const unsigned int &kMax,
	const unsigned int &tMax,
	const unsigned int &shaking_mode,
	const unsigned int &local_Search_mode,
	const unsigned int &update_xij_mode,
	const unsigned int &initial_mode,
	const unsigned int &time):
		VNS(directory, kMin, kMax, tMax, shaking_mode, 
			local_Search_mode, update_xij_mode, initial_mode, time),
		m_local_solution(m_locationNumber, 1)
{
	chrono::steady_clock::time_point timeStart = chrono::high_resolution_clock::now();
	initialSolution(m_incumbent_solution, m_bi, m_fi, m_dj, m_cij, m_sum_dj, m_initial_mode);

	unsigned int counter = 0, t = 0;
	bool stop = false;
	
	for (size_t i = 0; i != tMax && !stop; ++i)
	{
		// For all N_k Structures
		m_k = m_kMin - 1;
		while (m_k < m_kMax && !stop)
		{
			// Shake incumbent Solution
			m_perturbed_solution = shaking(m_incumbent_solution,
				m_bi, m_dj, m_cij, m_k, m_fx, m_shaking_mode);
			// Update Neighborhood Structure N_k(S')
			updateNeighborhoods(m_neighborhood_k, m_perturbed_solution, m_bi, 
				m_sum_dj, m_shaking_mode, m_k);
			// Generate Local Solution
			m_local_solution = localSearch(m_perturbed_solution, 
				m_neighborhood_k, m_k, m_fx, m_local_search_mode);
			// Move or not
			neighborhoodChange(m_incumbent_solution, m_local_solution, m_k, m_fx, m_best_fx, m_kMin);

			// Check if maximum COmputation Time is exeeded
			chrono::steady_clock::time_point bvns_step = chrono::high_resolution_clock::now();
			double time_end = (double)chrono::duration_cast<chrono::microseconds>(bvns_step - timeStart).count();
			if (time_end >= m_timeMax * 1000000.0)
			{
				cout << "Time Limit:" << m_timeMax << " exceeded." << endl;
				stop = true;
			}
		}
	}
	
	chrono::steady_clock::time_point timeEnd = chrono::high_resolution_clock::now();

	// Compute Optimal MODI Value
	updateXij(m_cij, m_dj, m_bi, m_incumbent_solution, m_update_xij_mode, m_best_fx);

	// Save Solution and Timestamps to File
	printFinalSolution(m_incumbent_solution, m_best_fx, m_kMax, m_tMax, m_shaking_mode, 
		m_local_search_mode, m_update_xij_mode, m_initial_mode, m_directory, timeStart, timeEnd);
}

vector<bool> BVNS::localSearch(
	const vector<bool> &perturbed_solution,
	const vector<vector<bool>> &neighborhood_k,
	const unsigned int &k,
	double &fx,
	const unsigned int &local_search_mode)
{
	if (local_search_mode == 0)
		return localSearchFirstImprovment(perturbed_solution, neighborhood_k, k, fx);
	else
		return localSearchBestImprovment(perturbed_solution, neighborhood_k, k, fx);
}







namespace ublas = boost::numeric::ublas;
using namespace std;

namespace t_simplex
{
	/* DECLARATION OF GLOBALS */
	double ** _tsC = NULL;				// Cost matrix
	double _tsMaxW;						// Maximum of all weights

										// MODI Method
	double transportSimplex(
		const unsigned &update_xij_mode,
		ublas::matrix<double> &custom_cij,
		TsFlow *flowTable,
		unsigned int *flowSize,
		TsSignature *facility,
		TsSignature *customer, 
		const double &sum_dj,
		const double &sum_bi,
		const vector <double > &dj)
	{
		int m = facility->n, n = customer->n; // Matrix Sizes
		int i = 0, j = 0;
		double totalCost = 0.0, diff = 0.0;
		int *P1 = NULL, *P2 = NULL;

		TsBasic *basics = NULL;					///Array of basic variables. 
		bool **isBasic = NULL;						//Flag matrix. isBasic[i][j] is true there is flow between source i and sink j
		TsBasic **facBasics = NULL;					//Array of pointers to the first basic variable in each row
		TsBasic **cusBasics = NULL;					//Array of pointers to the first basic variable in each column
		double *demand = facility->weights;						//Array of sink demands
		double *capacity = NULL;					//Array of source supplies
		bool flag = false;

		// Equalize source and sink weights.
		diff = sum_bi - sum_dj;
		if (diff > 0.0) n++;
		_tsMaxW = sum_bi > sum_dj ? sum_bi : sum_dj;

		basics = new TsBasic[m + n];
		isBasic = new bool*[m];
		for (i = 0; i < m; ++i)
			isBasic[i] = NULL;
		for (i = 0; i < m; ++i)
		{
			isBasic[i] = new bool[n];
			for (j = 0; j < n; ++j)
				isBasic[i][j] = 0;
		}
		facBasics = new TsBasic*[m];
		for (i = 0; i < m; ++i)
			facBasics[i] = NULL;
		cusBasics = new TsBasic*[n];
		for (i = 0; i < n; ++i)
			cusBasics[i] = NULL;

		// Compute the cost matrix
		_tsC = new double*[m];
		for (i = 0; i < m; ++i)
			_tsC[i] = NULL;

		for (i = 0, P1 = facility->features; i < m; ++i, P1++) 
		{
			_tsC[i] = new double[n];					
			for (j = 0, P2 = customer->features; j < n; ++j, P2++)
			{
				if (i == facility->n || j == customer->n) 
					_tsC[i][j] = 0;			
				else
					_tsC[i][j] = custom_cij(i, j);
			}
		}

		capacity = new double[m];					//init the source array
		for (i = 0; i < facility->n; ++i) 
			capacity[i] = facility->weights[i];

		demand = new double[n];					//init the sink array
		for (i = 0; i < customer->n; ++i) 
			demand[i] = customer->weights[i];

		if (n != customer->n)
			demand[customer->n] = diff;

		// Find the initail basic feasible solution. Use either _initRussel or _initVogel
		_initVogel(capacity, demand, basics, facBasics, cusBasics, isBasic, m, n);

		// Enter the main pivot loop
		totalCost = _pivot(basics, facBasics, cusBasics, isBasic, m, n, dj);

		// Fill the Flow data structure
		TsBasic * basicPtr = NULL;
		TsFlow * flowPtr = flowTable;
		unsigned int counter = 0;

		if (flowTable != NULL) {
			for (i = 0; i < m + n; ++i) {
				basicPtr = basics + i;

				if (basicPtr != NULL)
				{
					if (basicPtr->i >= 0 && basicPtr->i < facility->n
						&& basicPtr->j >= 0 && basicPtr->j < customer->n
						&& isBasic[basicPtr->i][basicPtr->j]
						&& basicPtr->val != 0.0
						&& counter < m + n - 1)
					{
						flowPtr->to = basicPtr->j;
						flowPtr->from = basicPtr->i;
						flowPtr->amount = basicPtr->val;
						flowPtr++;
						counter++;
					}
				}
			}
		}
		if (flowSize != NULL)
		{
			*flowSize = (int)(flowPtr - flowTable);
		}

		for (i = 0; i < m; i++)
		{
			delete[] isBasic[i];
			isBasic[i] = NULL;
		}
		delete[] isBasic;
		isBasic = NULL;

		for (i = 0; i < m; i++)
		{
			delete[] _tsC[i];
			_tsC[i] = NULL;
		}
		delete[] _tsC;
		_tsC = NULL;

		delete[] facBasics;
		facBasics = NULL;
		delete[] cusBasics;
		cusBasics = NULL;
		delete[] basics;
		basics = NULL;

		return totalCost;
	}

	/*
	Main pivot loop.
	Pivots until the system is optimal and return the optimal transportation cost.
	*/
	double _pivot(
		TsBasic * basics,
		TsBasic ** facBasics,
		TsBasic ** cusBasics,
		bool ** isBasic,
		int &m,
		int &n,
		const vector <double > &dj)
	{
		double * ui = NULL;
		double * vj = NULL;
		TsStone * stonePath = NULL;

		TsStone * spitra = NULL, *spitrb = NULL, *leaving = NULL;
		TsBasic * XP = NULL;
		TsBasic * basicsEnd = basics + m + n;
		TsBasic * entering = basicsEnd - 1;
		TsBasic dummyBasic;
		dummyBasic.i = -1;
		dummyBasic.j = 0;

		unsigned int i = 0, j = 0, lowI = 0, lowJ = 0, numPivots = 0;
		double objectiveValue = TSINFINITY, oldObjectiveValue = 0.0;
		double lowVal = 0.0;

		ui = new double[m];
		vj = new double[n];
		stonePath = new TsStone[m + n];

		while (1)
		{
			oldObjectiveValue = objectiveValue;
			objectiveValue = 0.0;

			for (XP = basics; XP != basicsEnd; XP++)
			{
				if (XP != entering)
				{
					//cout << endl << "i = " << XP->i << " j = " << XP->j << endl;
					if (XP->i >= 0 && XP->j >= 0 && XP->i < m && XP->j < dj.size()
						&& _tsC[XP->i][XP->j] != 0.0 && dj[XP->j] != 0.0)
					{
						objectiveValue += _tsC[XP->i][XP->j] * (XP->val / dj[XP->j]);
					}
				}
			}

			// Compute ui, vj
			stonePath[0].node = &dummyBasic;
			stonePath[0].prev = NULL;
			spitrb = _BFS(stonePath, facBasics, cusBasics, true);

			spitra = stonePath;
			vj[spitra->node->j] = 0;
			for (spitra++; spitra != spitrb; spitra++) {
				if (spitra->node->i == spitra->prev->node->i) {
					//node is in same row as parent
					vj[spitra->node->j] = _tsC[spitra->node->i][spitra->node->j] - ui[spitra->node->i];
				}
				else if (spitra->node->j == spitra->prev->node->j) {
					ui[spitra->node->i] = _tsC[spitra->node->i][spitra->node->j] - vj[spitra->node->j];
				}
			}

			// find Theta
			lowVal = 0.0;
			for (i = 0; i < m; ++i)
				for (j = 0; j < n; ++j)
					if (!isBasic[i][j] && _tsC[i][j] - ui[i] - vj[j] < lowVal)
					{
						lowVal = _tsC[i][j] - ui[i] - vj[j];
						lowI = i;
						lowJ = j;
					}

			if (lowVal >=  0.0 || (oldObjectiveValue - objectiveValue) < TSPIVOTLIMIT)
			{
				delete[] ui;
				delete[] vj;
				delete[] stonePath;
				//std::cout << "Pivots: " << numPivots << "\t";
				return objectiveValue;
			}

			// Add the entering variable to stone path
			entering->i = lowI;
			entering->j = lowJ;
			isBasic[lowI][lowJ] = 1;
			entering->val = 0;
			entering->nextFac = facBasics[lowI];
			if (facBasics[lowI] != NULL) facBasics[lowI]->prevFac = entering;
			entering->nextCus = cusBasics[lowJ];
			if (cusBasics[lowJ] != NULL) cusBasics[lowJ]->prevCus = entering;
			facBasics[lowI] = entering;
			entering->prevFac = facBasics[lowI];
			cusBasics[lowJ] = entering;
			entering->prevCus = cusBasics[lowJ];
			stonePath[0].node = entering;
			stonePath[0].prev = NULL;

			// Use breadth-first search to find a loop of basics.
			spitra = spitrb = _BFS(stonePath, facBasics, cusBasics);
			lowVal = TSINFINITY;
			bool add = false;

			// Find the lowest flow along the loop (leaving variable)
			do
			{
				if (!add && spitrb->node->val < lowVal)
				{
					leaving = spitrb;
					lowVal = spitrb->node->val;
				}
				add = !add;
			} while (spitrb = spitrb->prev);

			add = false;
			spitrb = spitra;

			// Alternately increase and decrease flow along the loop
			do
			{
				if (add)
					spitrb->node->val += lowVal;
				else
					spitrb->node->val -= lowVal;
				add = !add;
			} while (spitrb = spitrb->prev);

			i = leaving->node->i;
			j = leaving->node->j;
			isBasic[i][j] = 0;

			if (facBasics[i] == leaving->node)
			{
				facBasics[i] = leaving->node->nextFac;
				facBasics[i]->prevFac = NULL;
			}
			else
			{
				leaving->node->prevFac->nextFac = leaving->node->nextFac;
				if (leaving->node->nextFac != NULL)
					leaving->node->nextFac->prevFac = leaving->node->prevFac;
			}

			if (cusBasics[j] == leaving->node)
			{
				cusBasics[j] = leaving->node->nextCus;
				cusBasics[j]->prevCus = NULL;
			}
			else
			{
				leaving->node->prevCus->nextCus = leaving->node->nextCus;
				if (leaving->node->nextCus != NULL)
					leaving->node->nextCus->prevCus = leaving->node->prevCus;
			}

			entering = leaving->node;
			numPivots++;
		}
	}

	TsStone * _BFS(
		TsStone * stoneTree,
		TsBasic ** facBasics,
		TsBasic ** cusBasics,
		bool complete)
	{
		bool column = true;
		int jumpoffset = 0;
		TsBasic * bitr;
		TsStone * sitra = &stoneTree[0], *sitrb = &stoneTree[1];
		do {
			if (column)
			{
				for (bitr = cusBasics[sitra->node->j]; bitr != NULL; bitr = bitr->nextCus)
				{
					if (bitr != sitra->node) 
					{
						sitrb->node = bitr;
						sitrb->prev = sitra;
						sitrb++;
					}
				}
			}
			else 
			{
				for (bitr = facBasics[sitra->node->i]; bitr != NULL; bitr = bitr->nextFac)
				{
					if (bitr != sitra->node)
					{
						sitrb->node = bitr;
						sitrb->prev = sitra;
						sitrb++;
					}
				}
			}

			sitra++;
			if (sitra == sitrb) //no cycle found and no cycles in tree
				return sitra;

			if (sitra->node->i == sitra->prev->node->i)
				column = true;
			else
				column = false;

			// cycle found
			if (!complete && sitra->node->i == stoneTree[0].node->i
				&& sitra->node->j != stoneTree[0].node->j  && column == false)
				return sitra;
		} while (1);
	}

	/**********************
	Vogel's initialization method
	**********************/
	void _initVogel(
		double *S,
		double *D,
		TsBasic * basicsEnd,
		TsBasic ** facBasics,
		TsBasic ** cusBasics,
		bool ** isBasic,
		int m,
		int n)
	{
		int i, j;
		TsVogPen *srcPens = NULL;
		TsVogPen *snkPens = NULL;
		TsVogPen *pitra, *pitrb;  //iterators
		TsVogPen *maxPen;
		TsVogPen srcPenHead, snkPenHead;
		bool maxIsSrc;
		double lowVal;

		srcPens = new TsVogPen[m];
		snkPens = new TsVogPen[n];

		srcPenHead.next = pitra = srcPens;
		for (i = 0; i < m; i++)
		{
			pitra->i = i;
			pitra->next = pitra + 1;
			pitra->prev = pitra - 1;
			pitra->one = pitra->two = 0;
			pitra->oneCost = pitra->twoCost = TSINFINITY;
			pitra++;
		}
		(--pitra)->next = NULL;
		srcPens[0].prev = &srcPenHead;

		snkPenHead.next = pitra = snkPens;
		for (i = 0; i < n; i++) 
		{
			pitra->i = i;
			pitra->next = pitra + 1;
			pitra->prev = pitra - 1;
			pitra->one = pitra->two = 0;
			pitra->oneCost = pitra->twoCost = TSINFINITY;
			pitra++;
		}
		(--pitra)->next = NULL;
		snkPens[0].prev = &snkPenHead;


		for (pitra = srcPenHead.next, i = 0; pitra != NULL; pitra = pitra->next, i++)
		{
			for (pitrb = snkPenHead.next, j = 0; pitrb != NULL; pitrb = pitrb->next, j++)
			{
				//initialize Source Penalties;
				addPenalty(pitra, _tsC[i][j], j);
				addPenalty(pitrb, _tsC[i][j], i);
			}
		}

		while (srcPenHead.next != NULL && snkPenHead.next != NULL)
		{
			maxIsSrc = true;
			for (maxPen = pitra = srcPenHead.next; pitra != NULL; pitra = pitra->next)
				if ((pitra->twoCost - pitra->oneCost) > (maxPen->twoCost - maxPen->oneCost))
					maxPen = pitra;

			for (pitra = snkPenHead.next; pitra != NULL; pitra = pitra->next)
				if ((pitra->twoCost - pitra->oneCost) > (maxPen->twoCost - maxPen->oneCost))
				{
					maxPen = pitra;
					maxIsSrc = false;
				}

			if (maxIsSrc)
			{
				i = maxPen->i;
				j = maxPen->one;
			}
			else 
			{
				j = maxPen->i;
				i = maxPen->one;
			}

			if (D[j] - S[i] > _tsMaxW * TSEPSILON || (srcPenHead.next->next != NULL && fabs(S[i] - D[j]) < _tsMaxW * TSEPSILON))
			{
				//delete source
				lowVal = S[i];
				maxPen = srcPens + i;
				maxPen->prev->next = maxPen->next;
				if (maxPen->next != NULL)
					maxPen->next->prev = maxPen->prev;

				for (pitra = snkPenHead.next; pitra != NULL; pitra = pitra->next) {
					if (pitra->one == i || pitra->two == i) {
						pitra->oneCost = TSINFINITY;
						pitra->twoCost = TSINFINITY;
						for (pitrb = srcPenHead.next; pitrb != NULL; pitrb = pitrb->next)
							addPenalty(pitra, _tsC[pitrb->i][pitra->i], pitrb->i);
					}
				}
			}
			else 
			{
				//delete sink
				lowVal = D[j];
				maxPen = snkPens + j;
				maxPen->prev->next = maxPen->next;
				if (maxPen->next != NULL)
					maxPen->next->prev = maxPen->prev;

				for (pitra = srcPenHead.next; pitra != NULL; pitra = pitra->next)
				{
					if (pitra->one == j || pitra->two == j) {
						pitra->oneCost = TSINFINITY;
						pitra->twoCost = TSINFINITY;
						for (pitrb = snkPenHead.next; pitrb != NULL; pitrb = pitrb->next)
							addPenalty(pitra, _tsC[pitra->i][pitrb->i], pitrb->i);
					}
				}
			}

			S[i] -= lowVal;
			D[j] -= lowVal;

			isBasic[i][j] = 1;
			basicsEnd->val = lowVal;
			basicsEnd->i = i;
			basicsEnd->j = j;

			basicsEnd->nextCus = cusBasics[j];
			if (cusBasics[j] != NULL) cusBasics[j]->prevCus = basicsEnd;
			basicsEnd->nextFac = facBasics[i];
			if (facBasics[i] != NULL) facBasics[i]->prevFac = basicsEnd;

			facBasics[i] = basicsEnd;
			basicsEnd->prevCus = NULL;
			cusBasics[j] = basicsEnd;
			basicsEnd->prevFac = NULL;

			basicsEnd++;

		}
		delete[] srcPens;
		delete[] snkPens;
	}
}







namespace ublas = boost::numeric::ublas;
using namespace std;

namespace t_simplex
{
	/**********************
	Vogel's initialization method
	**********************/
	void _initVogel(
		vector<double> &bi,
		vector<double> dj,
		TsBasic * basicsEnd,
		TsBasic ** facBasics,
		TsBasic ** cusBasics,
		bool ** isBasic,
		int m,
		int n,
		ublas::matrix<double> &cij,
		double &_tsMaxW)
	{
		int i, j;
		TsVogPen *srcPens = NULL;
		TsVogPen *snkPens = NULL;
		TsVogPen *pitra = NULL, *pitrb = NULL;  //iterators
		TsVogPen *maxPen = NULL;
		TsVogPen srcPenHead, snkPenHead;
		bool maxIsSrc = false;
		double lowVal = 0.0;

		try
		{
			srcPens = new TsVogPen[m];
			snkPens = new TsVogPen[n];
		}
		catch (std::bad_alloc)
		{
			delete[] srcPens;
			delete[] snkPens;
			throw;
		}

		srcPenHead.next = pitra = srcPens;
		for (i = 0; i < m; i++)
		{
			pitra->i = i;
			pitra->next = pitra + 1;
			pitra->prev = pitra - 1;
			pitra->one = pitra->two = 0;
			pitra->oneCost = pitra->twoCost = TSINFINITY;
			pitra++;
		}
		(--pitra)->next = NULL;
		srcPens[0].prev = &srcPenHead;

		snkPenHead.next = pitra = snkPens;
		for (i = 0; i < n; i++)
		{
			pitra->i = i;
			pitra->next = pitra + 1;
			pitra->prev = pitra - 1;
			pitra->one = pitra->two = 0;
			pitra->oneCost = pitra->twoCost = TSINFINITY;
			pitra++;
		}
		(--pitra)->next = NULL;
		snkPens[0].prev = &snkPenHead;


		for (pitra = srcPenHead.next, i = 0; pitra != NULL; pitra = pitra->next, i++)
		{
			for (pitrb = snkPenHead.next, j = 0; pitrb != NULL; pitrb = pitrb->next, j++)
			{
				//initialize Source Penalties;
				addPenalty(pitra, cij(i, j), j);
				addPenalty(pitrb, cij(i, j), i);
			}
		}

		while (srcPenHead.next != NULL && snkPenHead.next != NULL)
		{
			maxIsSrc = true;
			for (maxPen = pitra = srcPenHead.next; pitra != NULL; pitra = pitra->next)
				if ((pitra->twoCost - pitra->oneCost) > (maxPen->twoCost - maxPen->oneCost))
					maxPen = pitra;

			for (pitra = snkPenHead.next; pitra != NULL; pitra = pitra->next)
				if ((pitra->twoCost - pitra->oneCost) > (maxPen->twoCost - maxPen->oneCost))
				{
					maxPen = pitra;
					maxIsSrc = false;
				}

			if (maxIsSrc)
			{
				i = maxPen->i;
				j = maxPen->one;
			}
			else
			{
				j = maxPen->i;
				i = maxPen->one;
			}

			if (dj[j] - bi[i] > _tsMaxW * TSEPSILON 
				|| (srcPenHead.next->next != NULL
				&& fabs(dj[i] - bi[j]) < _tsMaxW * TSEPSILON))
			{
				//delete source
				lowVal = bi[i];
				maxPen = srcPens + i;
				maxPen->prev->next = maxPen->next;
				if (maxPen->next != NULL)
					maxPen->next->prev = maxPen->prev;

				for (pitra = snkPenHead.next; pitra != NULL; pitra = pitra->next)
				{
					if (pitra->one == i || pitra->two == i)
					{
						pitra->oneCost = TSINFINITY;
						pitra->twoCost = TSINFINITY;
						for (pitrb = srcPenHead.next; pitrb != NULL; pitrb = pitrb->next)
							addPenalty(pitra, cij(pitrb->i, pitra->i), pitrb->i);
					}
				}
			}
			else
			{
				//delete sink
				lowVal = dj[j];
				maxPen = snkPens + j;
				maxPen->prev->next = maxPen->next;
				if (maxPen->next != NULL)
					maxPen->next->prev = maxPen->prev;

				for (pitra = srcPenHead.next; pitra != NULL; pitra = pitra->next)
				{
					if (pitra->one == j || pitra->two == j) {
						pitra->oneCost = TSINFINITY;
						pitra->twoCost = TSINFINITY;
						for (pitrb = snkPenHead.next; pitrb != NULL; pitrb = pitrb->next)
							addPenalty(pitra, cij(pitra->i, pitrb->i), pitrb->i);
					}
				}
			}

			bi[i] -= lowVal;
			dj[j] -= lowVal;

			isBasic[i][j] = 1;
			basicsEnd->val = lowVal;
			basicsEnd->i = i;
			basicsEnd->j = j;

			basicsEnd->nextCus = cusBasics[j];
			if (cusBasics[j] != NULL) cusBasics[j]->prevCus = basicsEnd;
			basicsEnd->nextFac = facBasics[i];
			if (facBasics[i] != NULL) facBasics[i]->prevFac = basicsEnd;

			facBasics[i] = basicsEnd;
			basicsEnd->prevCus = NULL;
			cusBasics[j] = basicsEnd;
			basicsEnd->prevFac = NULL;

			basicsEnd++;
		}

		delete[] srcPens;
		srcPens = NULL;
		delete[] snkPens;
		snkPens = NULL;
	}

	/**********************
	NW-Corner initialization method
	**********************/
	void _initNW(
		double *bi,
		double *dj,
		TsBasic * basicsEnd,
		TsBasic ** facBasics,
		TsBasic ** cusBasics,
		bool ** isBasic,
		int m,
		int n)
	{
		unsigned int i = 0, j = 0;
		double lowVal = 0.0;

		while (i < m && j < n)
		{
			// More capacity than Demand
			if (bi[i] >= dj[j] && dj[j] != 0.0)
			{
				lowVal = dj[j];
				bi[i] -= dj[j];
				dj[j] = 0.0;

				isBasic[i][j] = 1;
				basicsEnd->val = lowVal;
				basicsEnd->i = i;
				basicsEnd->j = j;

				basicsEnd->nextCus = cusBasics[j];
				if (cusBasics[j] != NULL) cusBasics[j]->prevCus = basicsEnd;
				basicsEnd->nextFac = facBasics[i];
				if (facBasics[i] != NULL) facBasics[i]->prevFac = basicsEnd;

				facBasics[i] = basicsEnd;
				basicsEnd->prevCus = NULL;
				cusBasics[j] = basicsEnd;
				basicsEnd->prevFac = NULL;

				basicsEnd++;

				// Test on Degeneration
				if (bi[i] == 0.0 && dj[j] == 0.0  && i < m - 1 && j < n - 1)
				{
					isBasic[i][j + 1] = 1;
					basicsEnd->val = lowVal;
					basicsEnd->i = i;
					basicsEnd->j = j + 1;

					basicsEnd->nextCus = cusBasics[j + 1];
					if (cusBasics[j + 1] != NULL) cusBasics[j + 1]->prevCus = basicsEnd;
					basicsEnd->nextFac = facBasics[i];
					if (facBasics[i] != NULL) facBasics[i]->prevFac = basicsEnd;

					facBasics[i] = basicsEnd;
					basicsEnd->prevCus = NULL;
					cusBasics[j + 1] = basicsEnd;
					basicsEnd->prevFac = NULL;

					basicsEnd++;

					i++;
					j++;
				}
				else
				{
					// Skip to next Customer
					j++;
				}
			}
			// Less Capacity than Demand
			else if (bi[i] < dj[j] && bi[i] != 0.0)
			{
				lowVal = bi[i];
				dj[j] -= bi[i];
				bi[i] = 0.0;

				isBasic[i][j] = 1;
				basicsEnd->val = lowVal;
				basicsEnd->i = i;
				basicsEnd->j = j;

				basicsEnd->nextCus = cusBasics[j];
				if (cusBasics[j] != NULL) cusBasics[j]->prevCus = basicsEnd;
				basicsEnd->nextFac = facBasics[i];
				if (facBasics[i] != NULL) facBasics[i]->prevFac = basicsEnd;

				facBasics[i] = basicsEnd;
				basicsEnd->prevCus = NULL;
				cusBasics[j] = basicsEnd;
				basicsEnd->prevFac = NULL;

				basicsEnd++;

				// Test on Degeneration
				if (bi[i] == 0.0 && dj[j] == 0.0 && i < m - 1 && j < n - 1)
				{
					isBasic[i][j+1] = 1;
					basicsEnd->val = lowVal;
					basicsEnd->i = i;
					basicsEnd->j = j+1;

					basicsEnd->nextCus = cusBasics[j+1];
					if (cusBasics[j+1] != NULL) cusBasics[j+1]->prevCus = basicsEnd;
					basicsEnd->nextFac = facBasics[i];
					if (facBasics[i] != NULL) facBasics[i]->prevFac = basicsEnd;

					facBasics[i] = basicsEnd;
					basicsEnd->prevCus = NULL;
					cusBasics[j+1] = basicsEnd;
					basicsEnd->prevFac = NULL;

					basicsEnd++;

					i++;
					j++;
				}
				else
				{
					// Skip to next Customer
					i++;
				}

			}
		}
	}

	/**********************
	LCM initialization method
	**********************/
	void _initLCM(
		double *bi,
		double *dj,
		TsBasic *basicsEnd,
		TsBasic **facBasics,
		TsBasic **cusBasics,
		bool ** isBasic,
		int m,
		int n,
		ublas::matrix<unsigned int> &cij_column_sorted,
		int *facilities)
	{
		unsigned int i = 0, j = 0;
		unsigned int indx_i = 0;
		double lowVal = 0.0;

		while (j < cij_column_sorted.size2())
		{
			indx_i = cij_column_sorted(i, j);

			// Can shift Capacity to fullfill Customer Demand
			if (bi[indx_i] != 0.0 && dj[j] != 0.0)
			{
				// More capacity than Demand
				if (bi[i] >= dj[j] && dj[j] != 0.0)
				{
					lowVal = dj[j];
					bi[i] -= dj[j];
					dj[j] = 0.0;

					isBasic[i][j] = 1;
					basicsEnd->val = lowVal;
					basicsEnd->i = i;
					basicsEnd->j = j;

					basicsEnd->nextCus = cusBasics[j];
					if (cusBasics[j] != NULL) cusBasics[j]->prevCus = basicsEnd;
					basicsEnd->nextFac = facBasics[i];
					if (facBasics[i] != NULL) facBasics[i]->prevFac = basicsEnd;

					facBasics[i] = basicsEnd;
					basicsEnd->prevCus = NULL;
					cusBasics[j] = basicsEnd;
					basicsEnd->prevFac = NULL;

					basicsEnd++;

					// Skip to next Customer
					j++;
					i = 0;
				}
				// Less Capacity than Demand
				else if (bi[i] < dj[j] && bi[i] != 0.0)
				{
					lowVal = bi[i];
					dj[j] -= bi[i];
					bi[i] = 0.0;

					isBasic[i][j] = 1;
					basicsEnd->val = lowVal;
					basicsEnd->i = i;
					basicsEnd->j = j;

					basicsEnd->nextCus = cusBasics[j];
					if (cusBasics[j] != NULL) cusBasics[j]->prevCus = basicsEnd;
					basicsEnd->nextFac = facBasics[i];
					if (facBasics[i] != NULL) facBasics[i]->prevFac = basicsEnd;

					facBasics[i] = basicsEnd;
					basicsEnd->prevCus = NULL;
					cusBasics[j] = basicsEnd;
					basicsEnd->prevFac = NULL;

					basicsEnd++;

					// Skip to next Customer
					i++;
				}
			}
		}

		// If its in the fictional Column
		if (n > cij_column_sorted.size2() && j == cij_column_sorted.size2())
		{
			for (i = 0; i != cij_column_sorted.size1(); ++i)
			{
				// If Capacity is not empty ship to customer
				if (bi[i] != 0.0)
				{
					lowVal = bi[i];
					bi[i] = 0.0;
					
					isBasic[i][j] = 1;
					basicsEnd->val = lowVal;
					basicsEnd->i = i;
					basicsEnd->j = j;

					basicsEnd->nextCus = cusBasics[j];
					if (cusBasics[j] != NULL) cusBasics[j]->prevCus = basicsEnd;
					basicsEnd->nextFac = facBasics[i];
					if (facBasics[i] != NULL) facBasics[i]->prevFac = basicsEnd;

					facBasics[i] = basicsEnd;
					basicsEnd->prevCus = NULL;
					cusBasics[j] = basicsEnd;
					basicsEnd->prevFac = NULL;

					basicsEnd++;
				}
			}
		}
	}
}







/*********************************************************************************/
/*                FIRST OR BEST IMPROVMENT LOCAL SEARCH                          */
/*********************************************************************************/

// Search First Improvment in N_k(S)
vector<bool> BVNS::localSearchFirstImprovment(
	const vector<bool> &perturbed_solution,
	const vector<vector<bool>> &neighborhood_k,
	const unsigned int &k,
	double &fx)
{
	int index = -1;
	unsigned int gap_mode = 0;
	double incumbent_value = fx;

	// Find first Imporvment in N_k(S')
	for (size_t i = 0; i != neighborhood_k.size(); ++i)
	{
		if (canUpdateXij(m_bi, neighborhood_k[i], m_sum_dj))
		{
			if (updateXij(m_cij, m_dj, m_bi, neighborhood_k[i], m_update_xij_mode, fx))
			{
				if (fx < incumbent_value)
				{
					index = i;
					break;
				}
			}
		}
	}
	// If Improvment was found
	if (index != -1)
		return neighborhood_k[index];
	// No Improvment found
	else
	{
		fx = incumbent_value;
		return perturbed_solution;
	}
}

// Search Best Improvment in N_k(S)
vector<bool> BVNS::localSearchBestImprovment(
	const vector<bool> &perturbed_solution,
	const vector<vector<bool>> &neighborhood_k,
	const unsigned int &k,
	double &fx)
{
	int index = -1;
	unsigned int gap_mode = 0;
	double incumbent_value = fx;
	double save_incumbent = fx;

	// Gets the best Solution in N_k(x) Neighborhood
	for (size_t i = 0; i != neighborhood_k.size(); ++i)
	{
		if (canUpdateXij(m_bi, neighborhood_k[i], m_sum_dj))
		{
			if (updateXij(m_cij, m_dj, m_bi, neighborhood_k[i], m_update_xij_mode, fx))
			{
				if (fx < incumbent_value)
				{
					index = i;
					incumbent_value = fx;
				}
			}
		}
	}

	// If Improvment was found
	if (index != -1)
	{
		fx = incumbent_value;
		return neighborhood_k[index];
	}
	// No Improvment found
	else
	{
		fx = save_incumbent;
		return perturbed_solution;
	}
}







using namespace t_simplex;
namespace ublas = boost::numeric::ublas;

/*********************************************************************************/
/*                            UPDATE XIJ PROCEDURES                              */
/*********************************************************************************/

// CFLP with Modi procedure (gets optimal solution in deterministic time)
bool VNS::updateXij(
	ublas::matrix<double> &cij,
	const vector<double> &dj,
	const vector<double> &bi,
	const vector<bool> &yi,
	const unsigned int &update_xij_mode,
	double &fx)
{
	if (!canUpdateXij(bi, yi, m_sum_dj))
		return false;

	unsigned int open_facilities = 0, i_out = 0, flow_vars = 0;
	double transportation_cost = 0.0, gap = 0.0, sum_bi = 0.0;
	bool check = false;

	open_facilities = accumulate(yi.begin(), yi.end(), 0);

	ublas::matrix<double> custom_cij(open_facilities, m_customerNumber, 0.0);

	int *customers = new int[m_customerNumber];
	double *demand = new double[m_customerNumber];
	int *facilities = new int[open_facilities];
	double *capacity = new double[open_facilities];

	for (size_t j = 0; j != cij.size2(); ++j)
	{
		customers[j] = j;
		demand[j] = dj[j];
	}

	for (size_t i = 0; i != cij.size1(); ++i)
	{
		if (yi[i] == 1)
		{
			facilities[i_out] = i;
			capacity[i_out] = bi[i];
			sum_bi += bi[i];

			ublas::matrix_row<ublas::matrix<double>> row_custom_cij(custom_cij, i_out);
			ublas::matrix_row<ublas::matrix<double>> row_cij(cij, i);
			row_custom_cij = row_cij;

			i_out++;
		}
	}

	// Signature of Facilities and Customers
	TsSignature *facility = new TsSignature(open_facilities, facilities, capacity);
	TsSignature *customer = new TsSignature(m_customerNumber, customers, demand);
	// Save Stepping Stone Path
	TsFlow *flow = new TsFlow[open_facilities + m_customerNumber - 1];

	// Result value
	transportation_cost = t_simplex::transportSimplex(update_xij_mode, custom_cij, flow, 
		&flow_vars, facility, customer ,m_sum_dj, sum_bi, dj);

	// f(x) and Return Value
	fx = f(yi, m_fi, transportation_cost);

	// Flow Xij for checkSolution
	m_flow_tpl.resize(flow_vars);
	for (size_t i = 0; i < flow_vars; ++i)
		m_flow_tpl[i] = make_tuple(facilities[flow[i].from], flow[i].to, flow[i].amount);

	check = checkSolution(yi, m_flow_tpl, flow_vars, cij, bi, dj, m_sum_dj);

	delete[] capacity;
	capacity = NULL;
	delete[] demand;
	demand = NULL;
	delete[] facilities;
	facilities = NULL;
	delete[] customers;
	customers = NULL;
	delete facility;
	facility = NULL;
	delete customer;
	customer = NULL;

	return check;
}








namespace ublas = boost::numeric::ublas;

/*********************************************************************************/
/*                                SHAKING PROCEDURES                             */
/*********************************************************************************/

// Creates Perturbed Solution S'
vector<bool> VNS::shakingKOperations(
	const vector<bool> &incumbent_solution,
	const vector<double> &bi,
	const vector<double> &dj,
	ublas::matrix<double> &cij,
	const double &k,
	double &fx)
{
	size_t last = 0, lastS = 0, counter = 0, i1 = 0, i2 = 0;
	vector<bool> perturbed_solution = incumbent_solution;
	double rnd = 0.0;

	vector<int> I_minus_S(incumbent_solution.size());
	vector<int> S(incumbent_solution.size());

	// I \ S := all closed Facilities
	last = 0;
	lastS = 0;
	for (size_t index = 0; index != perturbed_solution.size(); ++index)
		if (perturbed_solution[index] == false)
			I_minus_S[last++] = index;
		else
			S[lastS++] = index;
	I_minus_S.erase(I_minus_S.begin() + last, I_minus_S.end());
	S.erase(S.begin() + lastS, S.end());

	do
	{
		// Uniformed Random Value
		rnd = get_random_double(0.0, 1.0);

		// i1 element S
		if (S.size() != 0)
			i1 = S[select_randomly(S)];
		else
			i1 = 0;

		// i2 element I \ S
		if (I_minus_S.size() != 0)
			i2 = I_minus_S[select_randomly(I_minus_S)];
		else
			i2 = 0;

		// Drop
		if (rnd <= 0.2)
		{
			perturbed_solution[i1] = 0;
			I_minus_S.push_back(i1);
		}
		// Add
		else if (rnd >= 0.8)
		{
			perturbed_solution[i2] = 1;
			I_minus_S.erase(remove(
				I_minus_S.begin(), I_minus_S.end(), i2), I_minus_S.end());
		}
		// Swap
		else
		{
			perturbed_solution[i1] = 0;
			perturbed_solution[i2] = 1;

			I_minus_S.push_back(i1);
			I_minus_S.erase(remove(
				I_minus_S.begin(), I_minus_S.end(), i2), I_minus_S.end());
		}

		counter++;
	} while (counter < k); // Repeat k-Times

	// If its in Shaking-Phase and not in creating N_k(S) 
	if (fx != -1.0)
	{
		if (!canUpdateXij(bi, perturbed_solution, m_sum_dj))
		{
			fx = m_best_fx;
			return incumbent_solution;
		}

		// Check Solution if its not feasible return incumbent
		if (!updateXij(cij, dj, bi, perturbed_solution, m_update_xij_mode, fx))
		{
			fx = m_best_fx;
			return incumbent_solution;
		}

		return perturbed_solution;
	}

	return perturbed_solution;
}

// Creates Perturbed Solution S'
vector<bool> VNS::shakingKMaxOperations(
	const vector<bool> &incumbent_solution,
	const vector<double> &bi,
	const vector<double> &dj,
	ublas::matrix<double> &cij,
	const double &k,
	double &fx)
{
	size_t last = 0, lastS = 0, counter = 0, i1 = 0, i2 = 0;
	vector<bool> perturbed_solution = incumbent_solution;
	double rnd = 0.0;

	vector<int> I_minus_S(incumbent_solution.size());
	vector<int> S(incumbent_solution.size());

	// I \ S := all closed Facilities
	last = 0;
	lastS = 0;
	for (size_t index = 0; index < perturbed_solution.size(); ++index)
		if (perturbed_solution[index] == false)
			I_minus_S[last++] = index;
		else
			S[lastS++] = index;
	I_minus_S.erase(I_minus_S.begin() + last, I_minus_S.end());
	S.erase(S.begin() + lastS, S.end());

	do
	{
		// Uniformed Random Value
		rnd = get_random_double(0.0, 1.0);

		// i1 element S
		if (S.size() != 0)
			i1 = S[select_randomly(S)];
		else
			i1 = 0;

		// i2 element I \ S
		if (I_minus_S.size() != 0)
			i2 = I_minus_S[select_randomly(I_minus_S)];
		else
			i2 = 0;

		// Drop
		if (rnd <= 0.2)
		{
			perturbed_solution[i1] = 0;
			I_minus_S.push_back(i1);
		}
		// Add
		else if (rnd >= 0.8)
		{
			perturbed_solution[i2] = 1;
			I_minus_S.erase(remove(
				I_minus_S.begin(), I_minus_S.end(), i2), I_minus_S.end());
		}
		// Swap
		else
		{
			perturbed_solution[i1] = 0;
			perturbed_solution[i2] = 1;

			I_minus_S.push_back(i1);
			I_minus_S.erase(remove(
				I_minus_S.begin(), I_minus_S.end(), i2), I_minus_S.end());
		}		

		counter++;
	} while (counter < m_kMax); // Repeat k-Times

	// If its in Shaking-Phase and not in creating N_k(S) 
	if (fx != -1.0)
	{
		if (!canUpdateXij(bi, perturbed_solution, m_sum_dj))
		{
			fx = m_best_fx;
			return incumbent_solution;
		}

		// Check Solution if its not feasible return incumbent
		if (!updateXij(cij, dj, bi, perturbed_solution, m_update_xij_mode, fx))
		{
			fx = m_best_fx;
			return incumbent_solution;
		}

		return perturbed_solution;
	}

	return perturbed_solution;
}

// Shaking Method from Kratica et al.
vector<bool> VNS::shakingAssignments(
	const vector<bool> &incumbent_solution,
	const vector<double> &bi,
	const vector<double> &dj,
	ublas::matrix<double> &cij,
	const double &k,
	double &fx)
{
	unsigned int nh_mode = get_random_int(0, 2), i_out = 0, rand_indx = 0;
	double yi_sum = 0.0;
	size_t last = 0, lastS = 0, counter = 0, i1 = 0, i2 = 0;
	vector<bool> perturbed_solution = incumbent_solution;
	double rnd = 0.0;

	vector<int> I_minus_S(incumbent_solution.size());
	vector<int> S(incumbent_solution.size());

	// I \ S := all closed Facilities
	last = 0;
	lastS = 0;
	for (size_t index = 0; index < perturbed_solution.size(); ++index)
		if (perturbed_solution[index] == false)
			I_minus_S[last++] = index;
		else
			S[lastS++] = index;
	I_minus_S.erase(I_minus_S.begin() + last, I_minus_S.end());
	S.erase(S.begin() + lastS, S.end());
	
	// Swap
	if (nh_mode == 2)
	{
		for (size_t c = 0; c != k; ++c)
		{
			i1 = S[select_randomly(S)];
			i2 = I_minus_S[select_randomly(I_minus_S)];

			perturbed_solution[i1] = 0;
			perturbed_solution[i2] = 1;
		}
	}
	// Close k-Min/Max and open Random
	else
	{
		vector<double> assignments(m_locationNumber, 0);

		// Save all assignments
		for (size_t i = 0; i != m_flow_tpl.size(); ++i)
		{
			if (perturbed_solution[get<0>(m_flow_tpl[i])] == 1)
			{
				assignments[get<0>(m_flow_tpl[i])] += get<2>(m_flow_tpl[i]);
			}
		}

		// Index Sort Assignment list
		yi_sum = accumulate(incumbent_solution.begin(), incumbent_solution.end(), 0);
		vector<unsigned int> sorted_index;

		for (size_t indx = 0; indx != perturbed_solution.size(); ++indx)
		{
			if (perturbed_solution[indx] == 1)
			{
				sorted_index.push_back(indx);
			}
		}

		// sorted_index Index list
		sort(begin(sorted_index), end(sorted_index),
			[&](int i1, int i2) { return assignments[i1] < assignments[i2]; });

		if (nh_mode == 1)
		{
			for (size_t c = 0; c != k; ++c)
			{
				if (sorted_index.size() != 0 && c < perturbed_solution.size())
				{
					// Close k-Max
					if (c < sorted_index.size())
						perturbed_solution[sorted_index[c]] = 0;
					// Close k-Min
					if (sorted_index.size() - 1 - c >= 0)
						perturbed_solution[sorted_index[c]] = 0;
				}
			}
			for (size_t c = 0; c != k; ++c)
			{
				rand_indx = select_randomly(perturbed_solution);
				perturbed_solution[rand_indx] = 1;
			}
		}
		else
		{
			for (size_t c = 0; c != k; ++c)
			{
				if (sorted_index.size() != 0 && c < perturbed_solution.size())
				{
					// Close k-Min
					if (sorted_index.size() - 1 - c >= 0)
						perturbed_solution[sorted_index[c]] = 0;
				}
			}
			for (size_t c = 0; c != k; ++c)
			{
				rand_indx = select_randomly(perturbed_solution);
				perturbed_solution[rand_indx] = 1;
			}
		}	
	}

	// If its in Shaking-Phase and not in creating N_k(S) 
	if (fx != -1.0)
	{
		if (!canUpdateXij(bi, perturbed_solution, m_sum_dj))
		{
			fx = m_best_fx;
			return incumbent_solution;
		}

		// Check Solution if its not feasible return incumbent
		if (!updateXij(cij, dj, bi, perturbed_solution, m_update_xij_mode, fx))
		{
			fx = m_best_fx;
			return incumbent_solution;
		}

		return perturbed_solution;
	}

	return perturbed_solution;
}

// Modified Shaking Method from Kratica et al.
vector<bool> VNS::shakingCosts(
	const vector<bool> &incumbent_solution,
	const vector<double> &bi,
	const vector<double> &dj,
	ublas::matrix<double> &cij,
	const double &k,
	double &fx)
{
	unsigned int nh_mode = get_random_int(0, 2), i_out = 0, rand_indx = 0;
	double yi_sum = 0.0;
	size_t last = 0, lastS = 0, counter = 0, i1 = 0, i2 = 0;
	vector<bool> perturbed_solution = incumbent_solution;
	double rnd = 0.0;

	vector<int> I_minus_S(incumbent_solution.size());
	vector<int> S(incumbent_solution.size());

	// I \ S := all closed Facilities
	last = 0;
	lastS = 0;
	for (size_t index = 0; index < perturbed_solution.size(); ++index)
		if (perturbed_solution[index] == false)
			I_minus_S[last++] = index;
		else
			S[lastS++] = index;
	I_minus_S.erase(I_minus_S.begin() + last, I_minus_S.end());
	S.erase(S.begin() + lastS, S.end());

	// Normal Operations
	if (nh_mode == 1)
	{
		for (size_t c = 0; c != k; ++c)
		{
			i1 = S[select_randomly(S)];
			i2 = I_minus_S[select_randomly(I_minus_S)];

			perturbed_solution[i1] = 0;
			perturbed_solution[i2] = 1;
		}
	}
	// Close k-Min/Max and open Random
	else
	{
		vector<double> costs(m_locationNumber, 0);

		// Save all Costs per Facility
		for (size_t i = 0; i != cij.size1(); ++i)
		{
			ublas::matrix_row<ublas::matrix<double>> row_cij(cij, i);

			// For every xij link
			for (size_t f = 0; f != m_flow_tpl.size(); ++f)
			{
				if (get<0>(m_flow_tpl[f]) == i)
					costs[i] += row_cij[get<1>(m_flow_tpl[f])] * get<2>(m_flow_tpl[f]);
			}
		}

		// Index Sort Assignment list
		yi_sum = accumulate(perturbed_solution.begin(), perturbed_solution.end(), 0);
		vector<unsigned int> sorted_index;

		for (size_t indx = 0; indx != perturbed_solution.size(); ++indx)
		{
			if (perturbed_solution[indx] == 1)
			{
				sorted_index.push_back(indx);
			}
		}

		// sorted_index Index list
		sort(begin(sorted_index), end(sorted_index),
			[&](int i1, int i2) { return costs[i1] < costs[i2]; });

		if (nh_mode == 1)
		{
			for (size_t c = 0; c != k; ++c)
			{
				if (sorted_index.size() != 0 && c < perturbed_solution.size())
				{
					// Close k-Max
					if (c < sorted_index.size())
						perturbed_solution[sorted_index[c]] = 0;
				}
			}
			for (size_t c = 0; c != k; ++c)
			{
				do
				{
					rand_indx = select_randomly(perturbed_solution);
				} while (perturbed_solution[rand_indx] == 0);

				perturbed_solution[rand_indx] = 1;
			}

			for (size_t c = 0; c != k; ++c)
			{
				if (sorted_index.size() != 0 && c < perturbed_solution.size())
				{
					// Close k-Min
					if (sorted_index.size() - 1 - c >= 0)
						perturbed_solution[sorted_index[c]] = 0;
				}
			}
			for (size_t c = 0; c != k; ++c)
			{
				do
				{
					rand_indx = select_randomly(perturbed_solution);
				} while (perturbed_solution[rand_indx] == 0);

				perturbed_solution[rand_indx] = 1;
			}
		}
		// nh-gap_mode == 2
		else
		{
			for (size_t c = 0; c != k; ++c)
			{
				if (sorted_index.size() != 0 && c < perturbed_solution.size())
				{
					// Close k-Max
					if (c < sorted_index.size())
						perturbed_solution[sorted_index[c]] = 0;
				}
			}
			for (size_t c = 0; c != k; ++c)
			{
				rand_indx = select_randomly(perturbed_solution);
				perturbed_solution[rand_indx] = 1;
			}
		}	
	}

	// If its in Shaking-Phase and not in creating N_k(S) 
	if (fx != -1.0)
	{
		if (!canUpdateXij(bi, perturbed_solution, m_sum_dj))
		{
			fx = m_best_fx;
			return incumbent_solution;
		}

		// Check Solution if its not feasible return incumbent
		if (!updateXij(cij, dj, bi, perturbed_solution, m_update_xij_mode, fx))
		{
			fx = m_best_fx;
			return incumbent_solution;
		}

		return perturbed_solution;
	}

	return perturbed_solution;
}







namespace ublas = boost::numeric::ublas;

/*********************************************************************************/
/*                       UPDATE NEIGHBORHOOD PROCEDURES                          */
/*********************************************************************************/

// Creates or Updates Neighborhood Structure from Sol. S
void VNS::updateNeighborhoods(
	vector<vector<bool>> &neighborhood_k,
	const vector<bool> &perturbed_solution,
	const vector<double> &bi,
	const double &sum_dj,
	const unsigned int &shaking_mode,
	const double &k)
{
	double fx = -1;
	
	if (shaking_mode == 0)
	{
		// Iterate over all Neighborhood Solutions
		for (size_t n = 0; n < m_maxNH; ++n)
		{
			neighborhood_k[n] = shakingKOperations(perturbed_solution, bi, m_dj, m_cij, m_k, fx);
		}
	}
	else if(shaking_mode == 1)
	{
		// Iterate over all Neighborhood Solutions
		for (size_t n = 0; n < m_maxNH; ++n)
		{
			neighborhood_k[n] = shakingKMaxOperations(perturbed_solution, bi, m_dj, m_cij, m_k, fx);
		}
	}
	else if (shaking_mode == 2)
	{
		// Iterate over all Neighborhood Solutions
		for (size_t n = 0; n < m_maxNH; ++n)
		{
			neighborhood_k[n] = shakingAssignments(perturbed_solution, bi, m_dj, m_cij, m_k, fx);
		}
	}
	else
	{
		// Iterate over all Neighborhood Solutions
		for (size_t n = 0; n < m_maxNH; ++n)
		{
			neighborhood_k[n] = shakingCosts(perturbed_solution, bi, m_dj, m_cij, m_k, fx);
		}
	}
}





int main(int argc, char **argv)
{
	unsigned int kMin = 1, kMax = 0, tMax = 0, shaking_mode = 0, 
		update_neighborhood_mode = 0, local_search_mode = 0, update_xij_mode = 0, 
		stopping_critera = 0, init_mode = 0, rMax = 0, time = 0, rMax_mode = 0;

	vector<double> vals(10, 0.0);
	vector<double> vals_init(10, 0.0);

	string test_set_directory = "C:/Users/Jan/Dropbox/Bachelorarbeit/Programm/Testdaten/cap";
	BVNS *bvns = nullptr;

	vector<string> small_sets = {"10", "11","12", "13", "14", "15"};
	vector<string> medium_sets = { "16", "17", "18", "19", "20", "21" };
	vector<string> large_sets = { "a", "b", "c", "d", "e", "f" };
	vector<string> larger_sets = {"22","23","24","25","26","27","28","29",
		"30","31","32","33","34","35","36", "37", "38", "39", "40", "41", "42", "43", 
		"44", "45", "46", "47", "48", "49", "50", "51"};
	vector<string> all_sets = { "10", "11","12", "13", "14", "15", "16", "17", "18", "19", 
		"20", "21","a", "b", "c", "d", "e", "f", "22","23","24","25","26","27","28","29",
		"30","31","32","33","34","35","36", "37", "38", "39", "40", "41", "42", "43",
		"44", "45", "46", "47", "48", "49", "50", "51" };

	//vector<string> todo_sets = { "16", "a", "22", "28", "36", "44" };
	vector<string> todo_sets = { "22" };

	for (auto set : todo_sets)
	{
		cout << "Open Testset: cap" << set << endl;

		local_search_mode = 1;
		shaking_mode = 0;
		update_xij_mode = 1;
		init_mode = 1;
		time = 3000;
		tMax = 40;
		kMax = 3;

		bvns = new BVNS(test_set_directory + set, kMin, kMax, tMax,
			shaking_mode, local_search_mode, update_xij_mode, init_mode, time);

		cout << endl << "init = " << bvns->getInitialFx() << endl;
	}

	cout << endl << endl << "Finished Computation!" << endl;
	getchar();
	return 0;
}

// Init: RVNS, Local: Best, Xij: Vogel, Time Cap: 30s
/*init_mode = 1;
local_search_mode = 1;
update_xij_mode = 1;
time = 30;
tMax = 40;
kMax = 3;*/
