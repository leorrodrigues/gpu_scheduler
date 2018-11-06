#include "../../thirdparty/catch.hpp"
#include "../ahp.hpp"
#include <random>

int main(){

	AHP *ahp = new AHP();
	ahp->setHierarchy();
	// BUILD THE HIERARCHY
	// Focus
	auto f = ahp->hierarchy->addFocus("Choose the best car for the Jones Family");
	// Adding criterias
	// printf("FUCK 2\n");
	auto cost = ahp->hierarchy->addCriteria("Cost");
	float w1[4] = {1, 3, 7, 3};
	cost->setLeaf(false);
	// printf("FUCK 3\n");
	ahp->hierarchy->addEdge(f, cost, w1,4);
	auto safety = ahp->hierarchy->addCriteria("Safety");
	float w2[4] = {1 / 3., 1, 9., 1};
	// printf("FUCK 4\n");
	safety->setLeaf(true);
	ahp->hierarchy->addSheets(safety);
	ahp->hierarchy->addEdge(f, safety, w2,4);
	auto style = ahp->hierarchy->addCriteria("Style");
	// printf("FUCK 5\n");
	float w3[4] = {1 / 7., 1 / 9., 1, 1 / 7.};
	style->setLeaf(true);
	ahp->hierarchy->addSheets(style);
	ahp->hierarchy->addEdge(f, style, w3,4);
	auto capacity = ahp->hierarchy->addCriteria("Capacity");
	float w4[4] = {1 / 3., 1., 7, 1};
	capacity->setLeaf(false);
	ahp->hierarchy->addEdge(f, capacity, w4,4);
	// add cost childs
	auto purchase = ahp->hierarchy->addCriteria("Purchase Price");
	float w5[4] = {1, 2, 5, 3};
	purchase->setLeaf(true);
	ahp->hierarchy->addSheets(purchase);
	ahp->hierarchy->addEdge(cost, purchase, w5,4);
	auto fuelCosts = ahp->hierarchy->addCriteria("Fuel Costs");
	float w6[4] = {1 / 2., 1., 2, 2};
	fuelCosts->setLeaf(true);
	ahp->hierarchy->addSheets(fuelCosts);
	ahp->hierarchy->addEdge(cost, fuelCosts, w6,4);
	auto maintenanceCosts = ahp->hierarchy->addCriteria("Maintenance Costs");
	float w7[4] = {1 / 5., 1 / 2., 1, 1 / 2.};
	maintenanceCosts->setLeaf(true);
	ahp->hierarchy->addSheets(maintenanceCosts);
	ahp->hierarchy->addEdge(cost, maintenanceCosts, w7,4);
	auto resaleValue = ahp->hierarchy->addCriteria("Resale Value");
	float w8[4] = {1 / 3., 1 / 2., 2, 1};
	resaleValue->setLeaf(true);
	ahp->hierarchy->addSheets(resaleValue);
	ahp->hierarchy->addEdge(cost, resaleValue, w8,4);
	// Adding capacity childs
	auto cargoCapacity = ahp->hierarchy->addCriteria("Cargo Capacity");
	float w9[2] = {1, 1 / 5.};
	cargoCapacity->setLeaf(true);
	ahp->hierarchy->addSheets(cargoCapacity);
	ahp->hierarchy->addEdge(capacity, cargoCapacity, w9,2);
	auto passengerCapacity = ahp->hierarchy->addCriteria("Passenger Capacity");
	float w10[2] = {5, 1};
	passengerCapacity->setLeaf(true);
	ahp->hierarchy->addSheets(passengerCapacity);
	ahp->hierarchy->addEdge(capacity, passengerCapacity, w10,2);

	// Add Alternatives Resources
	// Add Alternatives
	auto a = ahp->hierarchy->addAlternative();
	a->setName("Accord Sedan");
	a = ahp->hierarchy->addAlternative();
	a->setName("Accord Hybrid");
	a = ahp->hierarchy->addAlternative();
	a->setName("Pilot SUV");
	a = ahp->hierarchy->addAlternative();
	a->setName("CR-V SUV");
	a = ahp->hierarchy->addAlternative();
	a->setName("Element SUV");
	a = ahp->hierarchy->addAlternative();
	a->setName("Odyssey Minivan");

	// adding edges through all sheets and alternatives
	ahp->hierarchy->addEdgeSheetsAlternatives();
	// Setting the alternatives priority to Purchase Price
	float pp1[6] = {1, 9, 9, 1, 1 / 2., 5};
	float pp2[6] = {1 / 9., 1, 1, 1 / 9., 1 / 9., 1 / 7.};
	float pp3[6] = {1 / 9., 1, 1, 1 / 9., 1 / 9., 1 / 7.};
	float pp4[6] = {1, 9, 9, 1, 1 / 2., 5};
	float pp5[6] = {2, 9, 9, 2, 1, 6};
	float pp6[6] = {1 / 5., 7, 7, 1 / 5., 1 / 6., 1};
	// Setting the alternatives priority to Safety
	float ps1[6] = {1, 1, 5, 7, 9, 1 / 3.};
	float ps2[6] = {1, 1, 5, 7, 9, 1 / 3.};
	float ps3[6] = {1 / 5., 1 / 5., 1, 2, 9, 1 / 8.};
	float ps4[6] = {1 / 7., 1 / 7., 1 / 2., 1, 2, 1 / 8.};
	float ps5[6] = {1 / 9., 1 / 9., 1 / 9., 1 / 2., 1, 1 / 9.};
	float ps6[6] = {3, 3, 8, 8, 9, 1};
	// Setting the alternatives priority to capacity
	float pc1[6] = {1, 1, 1 / 2., 1, 3, 1 / 2.};
	float pc2[6] = {1, 1, 1 / 2., 1, 3, 1 / 2.};
	float pc3[6] = {2, 2, 1, 2, 6, 1};
	float pc4[6] = {1, 1, 1 / 2., 1, 3, 1 / 2.};
	float pc5[6] = {1 / 3., 1 / 3., 1 / 6., 1 / 3., 1, 1 / 6.};
	float pc6[6] = {2, 2, 1, 2, 6, 1};
	// Setting the alternatives Priority to Fuel Costs

	float pfc1[6] = {1, 1 / (1.13), 1.41, 1.15, 1.24, 1.19};
	float pfc2[6] = {1.13, 1, 1.59, 1.3, 1.4, 1.35};
	float pfc3[6] = {1 / (1.141), 1 / (1.159), 1, 1 / (1.23), 1 / (1.14), (1 / 1.18)};
	float pfc4[6] = {1 / (1.15), 1 / (1.3), 1.23, 1, 1.08, 1.04};
	float pfc5[6] = {1 / (1.24), 1 / (1.4), 1.14, 1 / (1.08), 1, 1 / (1.04)};
	float pfc6[6] = {1 / (1.19), 1 / (1.35), 1.18, 1 / (1.04), 1.04, 1};
	// Setting the alternatives Priority to Resale Value
	float psv1[6] = {1, 3, 4, 1 / 2., 2, 2};
	float psv2[6] = {1 / 3., 1, 2, 1 / 5., 1, 1};
	float psv3[6] = {1 / 4., 1 / 2., 1, 1 / 6., 1 / 2., 1 / 2.};
	float psv4[6] = {2, 5, 6, 1, 4, 4};
	float psv5[6] = {1 / 2., 1, 2, 1 / 4., 1, 1};
	float psv6[6] = {1 / 2., 1, 2, 1 / 4., 1, 1};
	// Setting the alternatives Priority to Maintenance Costs
	float pm1[6] = {1, 1.5, 4, 4, 4, 5};
	float pm2[6] = {1 / (1.5), 1, 4, 4, 4, 5};
	float pm3[6] = {1 / 4., 1 / 4., 1, 1, 1.2, 1};
	float pm4[6] = {1 / 4., 1 / 4., 1, 1, 1, 3};
	float pm5[6] = {1 / 4., 1 / 4., 1 / (1.2), 1, 1, 2};
	float pm6[6] = {1 / 5., 1 / 5., 1, 1 / 3., 1 / 2., 1};
	// Setting the alternatives Priority to Style
	float pst1[6] = {1, 1, 7, 4, 9, 5};
	float pst2[6] = {1, 1, 7, 4, 9, 5};
	float pst3[6] = {1 / 7., 1 / 7., 1, 1 / 6., 3, 1 / 3.};
	float pst4[6] = {1 / 4., 1 / 4., 6, 1, 7, 5};
	float pst5[6] = {1 / 9., 1 / 9., 1 / 3., 1 / 7., 1, 1 / 5.};
	float pst6[6] = {1 / 5., 1 / 5., 3, 1 / 5., 5, 1};
	// Setting the alternatives Priority to Cargo Capacity
	float pcg1[6] = {1, 1, 1 / 2., 1 / 2., 1 / 2., 1 / 3.};
	float pcg2[6] = {1, 1, 1 / 2., 1 / 2., 1 / 2., 1 / 3.};
	float pcg3[6] = {2, 2, 1, 1, 1, 1 / 2.};
	float pcg4[6] = {2, 2, 1, 1, 1, 1 / 2.};
	float pcg5[6] = {2, 2, 1, 1, 1, 1 / 2.};
	float pcg6[6] = {3, 3, 2, 2, 2, 1};

	float* pp[6] = {pp1, pp2, pp3, pp4, pp5, pp6};
	float* ps[6] = {ps1, ps2, ps3, ps4, ps5, ps6};
	float* pc[6] = {pc1, pc2, pc3, pc4, pc5, pc6};
	float* pfc[6] = {pfc1, pfc2, pfc3, pfc4, pfc5, pfc6};
	float* psv[6] = {psv1, psv2, psv3, psv4, psv5, psv6};
	float* pm[6] = {pm1, pm2, pm3, pm4, pm5, pm6};
	float* pst[6] = {pst1, pst2, pst3, pst4, pst5, pst6};
	float* pcg[6] = {pcg1, pcg2, pcg3, pcg4, pcg5, pcg6};

	//    float*** alternativesWeights = {pp, ps, pc, pfc, psv, pm, pst, pcg}; ->
	//    This'll cause error on the PG answer,  because the vector is in wrong
	//    order.
	// CAUTION!!! The order of the alternatives Weights are HIGHLY dependent on
	// the order of hierarchy sheets.
	float** alternativesWeights[8] = {ps, pst, pp, pfc, pm, psv, pcg, pc};

	auto sheets = ahp->hierarchy->getSheets();
	int aSize = ahp->hierarchy->getAlternativesSize();
	for (int i = 0; i < ahp->hierarchy->getSheetsSize(); i++) {
		auto edges = sheets[i]->getEdges();
		for (int j = 0; j < sheets[i]->getSize(); j++) {
			edges[j]->setWeights(alternativesWeights[i][j], sheets[i]->getSize());
		}
	}

	ahp->synthesis();
	// ahp->consistency();

	auto pml = ahp->hierarchy->getFocus()->getPml();

	auto pg = ahp->hierarchy->getFocus()->getPg();

	delete(ahp);
}
