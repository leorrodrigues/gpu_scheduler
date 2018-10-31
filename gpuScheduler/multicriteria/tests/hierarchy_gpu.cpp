
#define CATCH_CONFIG_MAIN // This tells Catch to provide a main() - only do this
// in one cpp file
#include "../../thirdparty/catch.hpp"
#include "../ahpg.cuh"
#include <random>

typedef std::string VariablesType;
typedef float WeightType;

typedef std::vector<WeightType> Tv;
typedef std::vector<Tv> Tvv;
typedef std::vector<Tvv> Tvvv;

SCENARIO("Hierarchy can be created, insert criterias, focus and alternatives",
         "Hierarchy") {
	GIVEN("A new AHPG instance") {
		AHPG *ahpg = new AHPG();
		REQUIRE(ahpg->hierarchy->getCriterias().size() == 0);
		REQUIRE(ahpg->hierarchy->getSheetsCount() == 0);
		REQUIRE(ahpg->hierarchy->getFocus() == NULL);
		REQUIRE(ahpg->hierarchy->getResource()->mInt.size() == 0);
		REQUIRE(ahpg->hierarchy->getResource()->mIntSize == 0);
		REQUIRE(ahpg->hierarchy->getResource()->mWeight.size() == 0);
		REQUIRE(ahpg->hierarchy->getResource()->mWeightSize == 0);
		REQUIRE(ahpg->hierarchy->getResource()->mString.size() == 0);
		REQUIRE(ahpg->hierarchy->getResource()->mStringSize == 0);
		REQUIRE(ahpg->hierarchy->getResource()->mBoolSize == 0);
		REQUIRE(ahpg->hierarchy->getResource()->mBoolSize == 0);
		WHEN("The Focus are added") {
			auto str = "Objetivo Principal";
			auto focus = ahpg->hierarchy->addFocus(str);
			THEN("The focus object isn't null") {
				REQUIRE(ahpg->hierarchy->getFocus() != NULL);
			}
			THEN("The sent String to instanciate the Focus are now his name") {
				REQUIRE(ahpg->hierarchy->getFocus()->getName() == str);
			}
			THEN("All Edges and Matrix are NULL") {
				REQUIRE(ahpg->hierarchy->getFocus()->getMatrix() == NULL);
				REQUIRE(ahpg->hierarchy->getFocus()->getNormalizedMatrix() == NULL);
				REQUIRE(ahpg->hierarchy->getFocus()->getPml() == NULL);
				REQUIRE(ahpg->hierarchy->getFocus()->getPg() == NULL);
			}
			THEN("There are no edges") {
				REQUIRE(ahpg->hierarchy->getFocus()->edgesCount() == 0);
			}
			WHEN("Try to add new Focus") {
				THEN("The old Focus cant'n be overwritten") {
					auto newStr = "Novo Objetivo";
					auto newFocus = ahpg->hierarchy->addFocus(newStr);
					REQUIRE(newFocus != focus);
					REQUIRE(ahpg->hierarchy->getFocus() != newFocus);
					REQUIRE(ahpg->hierarchy->getFocus()->getName() != newStr);
					REQUIRE(ahpg->hierarchy->getFocus() == focus);
					REQUIRE(ahpg->hierarchy->getFocus()->getName() == str);
					REQUIRE(newFocus == NULL);
				}
			}
			WHEN("AHPG Matrix, Normalized Matrix, PML and PG are added") {
				WeightType **matrix, **normalizedMatrix, *pml, *pg;
				matrix = normalizedMatrix = NULL;
				pml = pg = NULL;
				WeightType lower_bound = 0;
				WeightType upper_bound = 10000;
				std::uniform_real_distribution<WeightType> unif(lower_bound,
				                                                upper_bound);
				std::default_random_engine re;
				matrix = new WeightType *[5];
				normalizedMatrix = new WeightType *[5];
				pml = new WeightType[5];
				pg = new WeightType[5];

				for (int i = 0; i < 5; i++) {
					pml[i] = unif(re);
					pg[i] = unif(re);
					matrix[i] = new WeightType[5];
					normalizedMatrix[i] = new WeightType[5];
					for (int j = 0; j < 5; j++) {
						matrix[i][j] = unif(re);
						normalizedMatrix[i][j] = unif(re);
					}
				}
				THEN("The new values are set in the Focus") {
					ahpg->hierarchy->getFocus()->setMatrix(matrix);
					ahpg->hierarchy->getFocus()->setNormalizedMatrix(normalizedMatrix);
					ahpg->hierarchy->getFocus()->setPml(pml);
					ahpg->hierarchy->getFocus()->setPg(pg);
					for (int i = 0; i < 5; i++) {
						REQUIRE(ahpg->hierarchy->getFocus()->getPml()[i] == pml[i]);
						REQUIRE(ahpg->hierarchy->getFocus()->getPg()[i] == pg[i]);
						for (int j = 0; j < 5; j++) {
							REQUIRE(ahpg->hierarchy->getFocus()->getMatrix()[i][j] ==
							        matrix[i][j]);
							REQUIRE(ahpg->hierarchy->getFocus()->getNormalizedMatrix()[i][j] ==
							        normalizedMatrix[i][j]);
						}
					}
				}
			}
		}
		WHEN("The Criteria are added") {
			int size = ahpg->hierarchy->getCriterias().size();
			THEN("The criteira are set with default values") {
				auto str1 = "criteria c1";
				auto c1 = ahpg->hierarchy->addCriteria(str1);
				REQUIRE(c1->getName() == str1);
				REQUIRE(c1->getEdges().size() == 0);
				REQUIRE(c1->getMatrix() == NULL);
				REQUIRE(c1->getNormalizedMatrix() == NULL);
				REQUIRE(c1->getPml() == NULL);
				REQUIRE(c1->getLeaf() == false);
				REQUIRE(c1->getActive() == true);
			}
			THEN("The criterias vector size grows") {
				auto str1 = "criteria c1";
				auto c1 = ahpg->hierarchy->addCriteria(str1);
				REQUIRE(ahpg->hierarchy->getCriterias().size() == size + 1);
			}
			THEN("Two criteiras with same name cannot be instantied") {
				auto str1 = "criteria c1";
				auto c1 = ahpg->hierarchy->addCriteria(str1);
				auto c2 = ahpg->hierarchy->addCriteria(str1);
				REQUIRE(c1->getName() == str1);
				REQUIRE(c1 != c2);
				REQUIRE(c2 == NULL);
			}
			THEN("Create two criterias and set and edge") {
				auto str1 = "criteria c1";
				auto str2 = "criteria c2";
				auto c1 = ahpg->hierarchy->addCriteria(str1);
				auto c2 = ahpg->hierarchy->addCriteria(str2);
				REQUIRE(c1 != c2);
				REQUIRE(c1->getName() == str1);
				REQUIRE(c2->getName() == str2);
				int edgesSize = c1->getEdges().size();
				ahpg->hierarchy->addEdge(c1, c2);
				REQUIRE(c1->getEdges().size() == edgesSize + 1);
				REQUIRE(c2->getParent() == c1);
			}
			THEN("Create a Focus and Criteria and set an edge") {
				auto f = ahpg->hierarchy->addFocus("focus");
				auto c = ahpg->hierarchy->addCriteria("criteria");
				int edgesSize = f->getEdges().size();
				ahpg->hierarchy->addEdge(f, c);
				REQUIRE(c->getOParent() == ahpg->hierarchy->getFocus());
				REQUIRE(f->getEdges().size() == edgesSize + 1);
			}
		}
		WHEN("The Alternative are added") {
			int size = ahpg->hierarchy->getAlternativesCount();
			auto alt = ahpg->hierarchy->addAlternative();
			THEN("The Alternative are set with default values") {
				auto re = alt->getResource();
				auto defRe = ahpg->hierarchy->getResource();
				REQUIRE(re->mInt == defRe->mInt);
				REQUIRE(re->mWeight == defRe->mWeight);
				REQUIRE(re->mString == defRe->mString);
				REQUIRE(re->mBool == defRe->mBool);
				REQUIRE(re->mIntSize == defRe->mIntSize);
				REQUIRE(re->mWeightSize == defRe->mWeightSize);
				REQUIRE(re->mStringSize == defRe->mStringSize);
				REQUIRE(re->mBoolSize == defRe->mBoolSize);
			}
			THEN("The alternatives vector size grows") {
				REQUIRE(ahpg->hierarchy->getAlternativesCount() == size + 1);
			}
		}
		WHEN("The Resources changes") {
			auto def = *ahpg->hierarchy->getResource();
			THEN("The int resource are added") {
				REQUIRE(ahpg->hierarchy->getResource()->mIntSize == def.mIntSize);
				ahpg->hierarchy->addResource("a", "int");
				REQUIRE(ahpg->hierarchy->getResource()->mIntSize == def.mIntSize + 1);
			}
			THEN("The bool resource are added") {
				REQUIRE(ahpg->hierarchy->getResource()->mBoolSize == def.mBoolSize);
				ahpg->hierarchy->addResource("a", "bool");
				REQUIRE(ahpg->hierarchy->getResource()->mBoolSize == def.mBoolSize + 1);
			}
			THEN("The std::String resource are added") {
				REQUIRE(ahpg->hierarchy->getResource()->mStringSize == def.mStringSize);
				ahpg->hierarchy->addResource("a", "string");
				REQUIRE(ahpg->hierarchy->getResource()->mStringSize ==
				        def.mStringSize + 1);
			}
			THEN("The WeightType resource are added") {
				REQUIRE(ahpg->hierarchy->getResource()->mWeightSize == def.mWeightSize);
				ahpg->hierarchy->addResource("a", "double");
				REQUIRE(ahpg->hierarchy->getResource()->mWeightSize ==
				        def.mWeightSize + 1);
			}
		}
		WHEN("The Hierarchy are constructed") {
			ahpg->hierarchy->addResource("name", "string");
			auto o = ahpg->hierarchy->addFocus("Qual host escolher?");
			auto c = ahpg->hierarchy->addCriteria("largura de banda");
			c->setLeaf(true);
			ahpg->hierarchy->addSheets(c);
			Tv weight = {1, 1, 1, 1};
			ahpg->hierarchy->addEdge(o, c, weight);
			c = ahpg->hierarchy->addCriteria("vCPU");
			c->setLeaf(true);
			ahpg->hierarchy->addSheets(c);
			ahpg->hierarchy->addEdge(o, c, weight);
			c = ahpg->hierarchy->addCriteria("QoS");
			c->setLeaf(true);
			ahpg->hierarchy->addSheets(c);
			ahpg->hierarchy->addEdge(o, c, weight);
			c = ahpg->hierarchy->addCriteria("Rede");
			c->setLeaf(false);
			ahpg->hierarchy->addEdge(o, c, weight);
			c = ahpg->hierarchy->addCriteria("L1");
			c->setLeaf(true);
			ahpg->hierarchy->addSheets(c);
			ahpg->hierarchy->addEdge(o, c, weight);
			c = ahpg->hierarchy->addCriteria("L2");
			c->setLeaf(true);
			ahpg->hierarchy->addSheets(c);
			ahpg->hierarchy->addEdge(o, c, weight);
			c = ahpg->hierarchy->addCriteria("L3");
			c->setLeaf(true);
			ahpg->hierarchy->addSheets(c);
			ahpg->hierarchy->addEdge(o, c, weight);
			c = ahpg->hierarchy->addCriteria("L4");
			c->setLeaf(true);
			ahpg->hierarchy->addSheets(c);
			ahpg->hierarchy->addEdge(o, c, weight);

			auto a = ahpg->hierarchy->addAlternative();
			a->setResource("name", "a1");
			a = ahpg->hierarchy->addAlternative();
			a->setResource("name", "a2");
			a = ahpg->hierarchy->addAlternative();
			a->setResource("name", "a3");
			a = ahpg->hierarchy->addAlternative();
			a->setResource("name", "a4");
			ahpg->hierarchy->addEdgeSheetsAlternatives();
			Tv weights(ahpg->hierarchy->getAlternativesCount(), 0.5);
			auto sheets = ahpg->hierarchy->getSheets();
			for (auto it = sheets.begin(); it != sheets.end(); it++) {
				auto ed = (*it)->getEdges();
				for (auto it2 = ed.begin(); it2 != ed.end(); it2++) {
					(*it2)->setWeights(weights);
				}
			}
			THEN("The hierarchy focus aren't NULL") {
				REQUIRE(ahpg->hierarchy->getFocus() != NULL);
			}
			THEN("The hierarchy criterias size aren't 0, the size corresponds to all "
			     "criterias (included sheets criterias)") {
				REQUIRE(ahpg->hierarchy->getCriterias().size() == 8);
			}
			THEN("The hierarchy sheets size aren't 0, his size corresponds to "
			     "criterias that have edges with alternatives") {
				REQUIRE(ahpg->hierarchy->getSheetsCount() == 7);
			}
			THEN("The hierarchy alternatives size aren't 0") {
				REQUIRE(ahpg->hierarchy->getAlternativesCount() == 4);
			}
			WHEN("Execute the ahpg synthesisG") {
				ahpg->synthesisG();
				THEN(
					"The hierarchy matrix, normalized matrix, pml and pg aren't NULL") {
					auto f = ahpg->hierarchy->getFocus();
					// REQUIRE(f->getMatrix() != NULL);
					// REQUIRE(f->getNormalizedMatrix() != NULL);
					REQUIRE(f->getPml() != NULL);
					REQUIRE(f->getPg() != NULL);
				}
			}
		}
	}
}

SCENARIO("Applying the AHPG example") {
	GIVEN("A Leader example") {
		AHPG *ahpg = new AHPG();
		// BUILD THE HIERARCHY
		// Focus
		auto f = ahpg->hierarchy->addFocus("Choose the most suitable leader");
		// Adding criterias
		auto c = ahpg->hierarchy->addCriteria("Experience");
		c->setLeaf(true);
		Tv fExperience = {1, 4, 3, 7};
		ahpg->hierarchy->addSheets(c);
		ahpg->hierarchy->addEdge(f, c, fExperience);
		auto c2 = ahpg->hierarchy->addCriteria("Education");
		c2->setLeaf(true);
		Tv fEducation = {1 / 4., 1, 1 / 3., 3};
		ahpg->hierarchy->addSheets(c2);
		ahpg->hierarchy->addEdge(f, c2, fEducation);
		c = ahpg->hierarchy->addCriteria("Charisma");
		c->setLeaf(true);
		Tv fCharisma = {1 / 3., 3, 1, 5};
		ahpg->hierarchy->addSheets(c);
		ahpg->hierarchy->addEdge(f, c, fCharisma);
		c = ahpg->hierarchy->addCriteria("Age");
		c->setLeaf(true);
		Tv fAge = {1 / 7., 1 / 3., 1 / 5., 1};
		ahpg->hierarchy->addSheets(c);
		ahpg->hierarchy->addEdge(f, c, fAge);
		// Add Alternatives Resources
		ahpg->hierarchy->addResource("name", "string");
		// Add Alternatives
		auto a = ahpg->hierarchy->addAlternative();
		a->setResource("name", "tom");
		a = ahpg->hierarchy->addAlternative();
		a->setResource("name", "Dick");
		a = ahpg->hierarchy->addAlternative();
		a->setResource("name", "Harry");
		// adding edges through all sheets and alternatives
		ahpg->hierarchy->addEdgeSheetsAlternatives();
		// First Criteria to Alternatives
		Tv w11 = {1, 1 / 4., 4};
		Tv w12 = {4, 1, 9};
		Tv w13 = {1 / 4., 1 / 9., 1};
		Tvv experienceWeights = {w11, w12, w13};
		// Second Criteira to Alternatives
		Tv w21 = {1, 3, 1 / 5.};
		Tv w22 = {1 / 3., 1, 1 / 7.};
		Tv w23 = {5, 7, 1};
		Tvv educationWeights = {w21, w22, w23};
		// Third Criteira to Alternatives
		Tv w31 = {1, 5, 9};
		Tv w32 = {1 / 5., 1, 4};
		Tv w33 = {1 / 9., 1 / 4., 1};
		Tvv charismaWeights = {w31, w32, w33};
		// Fourth Criteira to Alternatives
		Tv w41 = {1, 1 / 3., 5};
		Tv w42 = {3, 1, 9};
		Tv w43 = {1 / 5., 1 / 9., 1};
		Tvv ageWeights = {w41, w42, w43};

		Tvvv alternativesWeights = {experienceWeights, educationWeights,
			                    charismaWeights, ageWeights};

		auto sheets = ahpg->hierarchy->getSheets();
		int aSize = ahpg->hierarchy->getAlternativesCount();
		for (int i = 0; i < sheets.size(); i++) {
			auto edges = sheets[i]->getEdges();
			for (int j = 0; j < edges.size(); j++) {
				edges[j]->setWeights(alternativesWeights[i][j]);
			}
		}
		// run the synthesisG and consistencyG ahpg functions

		ahpg->synthesisG();

		// ahpg->consistencyG();
		THEN("check the local priority") {
			auto pml = ahpg->hierarchy->getFocus()->getPml();
			REQUIRE(pml[0] + pml[1] + pml[2] + pml[3] == 1);
		}

		THEN("check the global priority") {
			auto pg = ahpg->hierarchy->getFocus()->getPg();
			REQUIRE(pg[0] < pg[1]);
			REQUIRE(pg[2] < pg[0]);
			REQUIRE(pg[0] + pg[1] + pg[2] == 1);
		}
	}

	GIVEN("A Car example") {

		AHPG *ahpg = new AHPG();
		// BUILD THE HIERARCHY
		// Focus
		auto f =
			ahpg->hierarchy->addFocus("Choose the best car for the Jones Family");
		// Adding criterias
		auto cost = ahpg->hierarchy->addCriteria("Cost");
		Tv w1 = {1, 3, 7, 3};
		cost->setLeaf(false);
		ahpg->hierarchy->addEdge(f, cost, w1);
		auto safety = ahpg->hierarchy->addCriteria("Safety");
		Tv w2 = {1 / 3., 1, 9., 1};
		safety->setLeaf(true);
		ahpg->hierarchy->addSheets(safety);
		ahpg->hierarchy->addEdge(f, safety, w2);
		auto style = ahpg->hierarchy->addCriteria("Style");
		Tv w3 = {1 / 7., 1 / 9., 1, 1 / 7.};
		style->setLeaf(true);
		ahpg->hierarchy->addSheets(style);
		ahpg->hierarchy->addEdge(f, style, w3);
		auto capacity = ahpg->hierarchy->addCriteria("Capacity");
		Tv w4 = {1 / 3., 1., 7, 1};
		capacity->setLeaf(false);
		ahpg->hierarchy->addEdge(f, capacity, w4);
		// add cost childs
		auto purchase = ahpg->hierarchy->addCriteria("Purchase Price");
		Tv w5 = {1, 2, 5, 3};
		purchase->setLeaf(true);
		ahpg->hierarchy->addSheets(purchase);
		ahpg->hierarchy->addEdge(cost, purchase, w5);
		auto fuelCosts = ahpg->hierarchy->addCriteria("Fuel Costs");
		Tv w6 = {1 / 2., 1., 2, 2};
		fuelCosts->setLeaf(true);
		ahpg->hierarchy->addSheets(fuelCosts);
		ahpg->hierarchy->addEdge(cost, fuelCosts, w6);
		auto maintenanceCosts = ahpg->hierarchy->addCriteria("Maintenance Costs");
		Tv w7 = {1 / 5., 1 / 2., 1, 1 / 2.};
		maintenanceCosts->setLeaf(true);
		ahpg->hierarchy->addSheets(maintenanceCosts);
		ahpg->hierarchy->addEdge(cost, maintenanceCosts, w7);
		auto resaleValue = ahpg->hierarchy->addCriteria("Resale Value");
		Tv w8 = {1 / 3., 1 / 2., 2, 1};
		resaleValue->setLeaf(true);
		ahpg->hierarchy->addSheets(resaleValue);
		ahpg->hierarchy->addEdge(cost, resaleValue, w8);
		// Adding capacity childs
		auto cargoCapacity = ahpg->hierarchy->addCriteria("Cargo Capacity");
		Tv w9 = {1, 1 / 5.};
		cargoCapacity->setLeaf(true);
		ahpg->hierarchy->addSheets(cargoCapacity);
		ahpg->hierarchy->addEdge(capacity, cargoCapacity, w9);
		auto passengerCapacity = ahpg->hierarchy->addCriteria("Passenger Capacity");
		Tv w10 = {5, 1};
		passengerCapacity->setLeaf(true);
		ahpg->hierarchy->addSheets(passengerCapacity);
		ahpg->hierarchy->addEdge(capacity, passengerCapacity, w10);

		// Add Alternatives Resources
		ahpg->hierarchy->addResource("name", "string");
		// Add Alternatives
		auto a = ahpg->hierarchy->addAlternative();
		a->setResource("name", "Accord Sedan");
		a = ahpg->hierarchy->addAlternative();
		a->setResource("name", "Accord Hybrid");
		a = ahpg->hierarchy->addAlternative();
		a->setResource("name", "Pilot SUV");
		a = ahpg->hierarchy->addAlternative();
		a->setResource("name", "CR-V SUV");
		a = ahpg->hierarchy->addAlternative();
		a->setResource("name", "Element SUV");
		a = ahpg->hierarchy->addAlternative();
		a->setResource("name", "Odyssey Minivan");

		// adding edges through all sheets and alternatives
		ahpg->hierarchy->addEdgeSheetsAlternatives();
		// Setting the alternatives priority to Purchase Price
		Tv pp1 = {1, 9, 9, 1, 1 / 2., 5};
		Tv pp2 = {1 / 9., 1, 1, 1 / 9., 1 / 9., 1 / 7.};
		Tv pp3 = {1 / 9., 1, 1, 1 / 9., 1 / 9., 1 / 7.};
		Tv pp4 = {1, 9, 9, 1, 1 / 2., 5};
		Tv pp5 = {2, 9, 9, 2, 1, 6};
		Tv pp6 = {1 / 5., 7, 7, 1 / 5., 1 / 6., 1};
		// Setting the alternatives priority to Safety
		Tv ps1 = {1, 1, 5, 7, 9, 1 / 3.};
		Tv ps2 = {1, 1, 5, 7, 9, 1 / 3.};
		Tv ps3 = {1 / 5., 1 / 5., 1, 2, 9, 1 / 8.};
		Tv ps4 = {1 / 7., 1 / 7., 1 / 2., 1, 2, 1 / 8.};
		Tv ps5 = {1 / 9., 1 / 9., 1 / 9., 1 / 2., 1, 1 / 9.};
		Tv ps6 = {3, 3, 8, 8, 9, 1};
		// Setting the alternatives priority to capacity
		Tv pc1 = {1, 1, 1 / 2., 1, 3, 1 / 2.};
		Tv pc2 = {1, 1, 1 / 2., 1, 3, 1 / 2.};
		Tv pc3 = {2, 2, 1, 2, 6, 1};
		Tv pc4 = {1, 1, 1 / 2., 1, 3, 1 / 2.};
		Tv pc5 = {1 / 3., 1 / 3., 1 / 6., 1 / 3., 1, 1 / 6.};
		Tv pc6 = {2, 2, 1, 2, 6, 1};
		// Setting the alternatives Priority to Fuel Costs

		Tv pfc1 = {1, 1 / (1.13), 1.41, 1.15, 1.24, 1.19};
		Tv pfc2 = {1.13, 1, 1.59, 1.3, 1.4, 1.35};
		Tv pfc3 = {1 / (1.141), 1 / (1.159), 1, 1 / (1.23), 1 / (1.14), (1 / 1.18)};
		Tv pfc4 = {1 / (1.15), 1 / (1.3), 1.23, 1, 1.08, 1.04};
		Tv pfc5 = {1 / (1.24), 1 / (1.4), 1.14, 1 / (1.08), 1, 1 / (1.04)};
		Tv pfc6 = {1 / (1.19), 1 / (1.35), 1.18, 1 / (1.04), 1.04, 1};
		// Setting the alternatives Priority to Resale Value
		Tv psv1 = {1, 3, 4, 1 / 2., 2, 2};
		Tv psv2 = {1 / 3., 1, 2, 1 / 5., 1, 1};
		Tv psv3 = {1 / 4., 1 / 2., 1, 1 / 6., 1 / 2., 1 / 2.};
		Tv psv4 = {2, 5, 6, 1, 4, 4};
		Tv psv5 = {1 / 2., 1, 2, 1 / 4., 1, 1};
		Tv psv6 = {1 / 2., 1, 2, 1 / 4., 1, 1};
		// Setting the alternatives Priority to Maintenance Costs
		Tv pm1 = {1, 1.5, 4, 4, 4, 5};
		Tv pm2 = {1 / (1.5), 1, 4, 4, 4, 5};
		Tv pm3 = {1 / 4., 1 / 4., 1, 1, 1.2, 1};
		Tv pm4 = {1 / 4., 1 / 4., 1, 1, 1, 3};
		Tv pm5 = {1 / 4., 1 / 4., 1 / (1.2), 1, 1, 2};
		Tv pm6 = {1 / 5., 1 / 5., 1, 1 / 3., 1 / 2., 1};
		// Setting the alternatives Priority to Style
		Tv pst1 = {1, 1, 7, 4, 9, 5};
		Tv pst2 = {1, 1, 7, 4, 9, 5};
		Tv pst3 = {1 / 7., 1 / 7., 1, 1 / 6., 3, 1 / 3.};
		Tv pst4 = {1 / 4., 1 / 4., 6, 1, 7, 5};
		Tv pst5 = {1 / 9., 1 / 9., 1 / 3., 1 / 7., 1, 1 / 5.};
		Tv pst6 = {1 / 5., 1 / 5., 3, 1 / 5., 5, 1};
		// Setting the alternatives Priority to Cargo Capacity
		Tv pcg1 = {1, 1, 1 / 2., 1 / 2., 1 / 2., 1 / 3.};
		Tv pcg2 = {1, 1, 1 / 2., 1 / 2., 1 / 2., 1 / 3.};
		Tv pcg3 = {2, 2, 1, 1, 1, 1 / 2.};
		Tv pcg4 = {2, 2, 1, 1, 1, 1 / 2.};
		Tv pcg5 = {2, 2, 1, 1, 1, 1 / 2.};
		Tv pcg6 = {3, 3, 2, 2, 2, 1};

		Tvv pp = {pp1, pp2, pp3, pp4, pp5, pp6};
		Tvv ps = {ps1, ps2, ps3, ps4, ps5, ps6};
		Tvv pc = {pc1, pc2, pc3, pc4, pc5, pc6};
		Tvv pfc = {pfc1, pfc2, pfc3, pfc4, pfc5, pfc6};
		Tvv psv = {psv1, psv2, psv3, psv4, psv5, psv6};
		Tvv pm = {pm1, pm2, pm3, pm4, pm5, pm6};
		Tvv pst = {pst1, pst2, pst3, pst4, pst5, pst6};
		Tvv pcg = {pcg1, pcg2, pcg3, pcg4, pcg5, pcg6};

		//    Tvvv alternativesWeights = {pp, ps, pc, pfc, psv, pm, pst, pcg}; ->
		//    This'll cause error on the PG answer,  because the vector is in wrong
		//    order.
		// CAUTION!!! The order of the alternatives Weights are HIGHLY dependent on
		// the order of hierarchy sheets.
		Tvvv alternativesWeights = {ps, pst, pp, pfc, pm, psv, pcg, pc};

		auto sheets = ahpg->hierarchy->getSheets();
		int aSize = ahpg->hierarchy->getAlternativesCount();
		for (int i = 0; i < sheets.size(); i++) {
			auto edges = sheets[i]->getEdges();
			for (int j = 0; j < edges.size(); j++) {
				edges[j]->setWeights(alternativesWeights[i][j]);
			}
		}

		ahpg->synthesisG();
		// ahpg->consistencyG();

		THEN("Check the local priority") {
			auto pml = ahpg->hierarchy->getFocus()->getPml();

			REQUIRE(pml[0] + pml[1] + pml[2] + pml[3] == 1);
		}
		THEN("Check the global priority") {
			auto pg = ahpg->hierarchy->getFocus()->getPg();
			REQUIRE(pg[0] + pg[1] + pg[2] + pg[3] + pg[4] + pg[5] == 1);
			REQUIRE(pg[5] > pg[0]);
			REQUIRE(pg[0] > pg[3]);
			REQUIRE(pg[3] > pg[1]);
			REQUIRE(pg[1] > pg[4]);
			REQUIRE(pg[4] > pg[2]);
		}
	}
}
