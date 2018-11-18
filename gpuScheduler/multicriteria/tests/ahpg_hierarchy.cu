
#define CATCH_CONFIG_MAIN // This tells Catch to provide a main() - only do this
// in one cpp file
#include "../../thirdparty/catch.hpp"
#include "../ahpg.cuh"
#include <random>
//
// SCENARIO("Hierarchy can be created, insert criterias, focus and alternatives",
//          "Hierarchy") {
//      GIVEN("A new AHP instance") {
//              AHPG* ahpg = new AHPG();
//              REQUIRE(ahpg->hierarchy == NULL);
//              ahpg->setHierarchyG();
//              REQUIRE(ahpg->hierarchy->getCriteriasSize() == 0);
//              REQUIRE(ahpg->hierarchy->getSheetsSize() == 0);
//              REQUIRE(ahpg->hierarchy->getFocus() == NULL);
//              REQUIRE(ahpg->hierarchy->getResource()->getDataSize() == 0);
//              WHEN("The Focus are added") {
//                      auto str = "Objetivo Principal";
//                      auto focus = ahpg->hierarchy->addFocus(str);
//                      THEN("The focus object isn't null") {
//                              REQUIRE(ahpg->hierarchy->getFocus() != NULL);
//                      }
//                      THEN("The sent String to instanciate the Focus are now his name") {
//                              REQUIRE(std::string(ahpg->hierarchy->getFocus()->getName()) == str);
//                      }
//                      THEN("All Edges and Matrix are NULL") {
//                              REQUIRE(ahpg->hierarchy->getFocus()->getMatrix() == NULL);
//                              REQUIRE(ahpg->hierarchy->getFocus()->getNormalizedMatrix() == NULL);
//                              REQUIRE(ahpg->hierarchy->getFocus()->getPml() == NULL);
//                              REQUIRE(ahpg->hierarchy->getFocus()->getPg() == NULL);
//                      }
//                      THEN("There are no edges") {
//                              REQUIRE(ahpg->hierarchy->getFocus()->getSize() == 0);
//                      }
//                      WHEN("Try to add new Focus") {
//                              THEN("The old Focus cant'n be overwritten") {
//                                      auto newStr = "Novo Objetivo";
//                                      auto newFocus = ahpg->hierarchy->addFocus(newStr);
//                                      REQUIRE(newFocus != focus);
//                                      REQUIRE(ahpg->hierarchy->getFocus() != newFocus);
//                                      REQUIRE(std::string(ahpg->hierarchy->getFocus()->getName()) != newStr);
//                                      REQUIRE(ahpg->hierarchy->getFocus() == focus);
//                                      REQUIRE(std::string(ahpg->hierarchy->getFocus()->getName()) == str);
//                                      REQUIRE(newFocus == NULL);
//                              }
//                      }
//                      WHEN("AHP Matrix, Normalized Matrix, PML and PG are added") {
//                              float* matrix, *normalizedMatrix, *pml, *pg;
//                              matrix = normalizedMatrix = NULL;
//                              pml = pg = NULL;
//                              float lower_bound = 0;
//                              float upper_bound = 10000;
//                              std::uniform_real_distribution<float> unif(lower_bound,
//                                                                         upper_bound);
//                              std::default_random_engine re;
//                              matrix = new float [25];
//                              normalizedMatrix = new float [25];
//                              pml = new float[5];
//                              pg = new float[5];
//
//                              for (int i = 0; i < 5; i++) {
//                                      pml[i] = unif(re);
//                                      pg[i] = unif(re);
//                                      for (int j = 0; j < 5; j++) {
//                                              matrix[i*5+j] = unif(re);
//                                              normalizedMatrix[i*5+j] = unif(re);
//                                      }
//                              }
//                              THEN("The new values are set in the Focus") {
//                                      ahpg->hierarchy->getFocus()->setSize(5);
//                                      ahpg->hierarchy->getFocus()->setMatrix(matrix);
//                                      ahpg->hierarchy->getFocus()->setNormalizedMatrix(normalizedMatrix);
//                                      ahpg->hierarchy->getFocus()->setPml(pml);
//                                      ahpg->hierarchy->getFocus()->setPg(pg);
//                                      for (int i = 0; i < 5; i++) {
//                                              REQUIRE(ahpg->hierarchy->getFocus()->getSize() == 5);
//                                              REQUIRE(ahpg->hierarchy->getFocus()->getPml()[i] == pml[i]);
//                                              REQUIRE(ahpg->hierarchy->getFocus()->getPg()[i] == pg[i]);
//                                              for (int j = 0; j < 5; j++) {
//                                                      REQUIRE(ahpg->hierarchy->getFocus()->getMatrix()[i*5+j] ==
//                                                              matrix[i*5+j]);
//                                                      REQUIRE(ahpg->hierarchy->getFocus()->getNormalizedMatrix()[i*5+j] ==
//                                                              normalizedMatrix[i*5+j]);
//                                              }
//                                      }
//                              }
//                      }
//              }
//              WHEN("The Criteria are added") {
//                      int size = ahpg->hierarchy->getCriteriasSize();
//                      THEN("The criteira are set with default values") {
//                              auto str1 = "criteria c1";
//                              Node* c1 = ahpg->hierarchy->addCriteria(str1);
//                              REQUIRE(std::string(c1->getName()) == str1);
//                              REQUIRE(c1->getSize() == 0);
//                              REQUIRE(c1->getMatrix() == NULL);
//                              REQUIRE(c1->getNormalizedMatrix() == NULL);
//                              REQUIRE(c1->getPml() == NULL);
//                              REQUIRE(c1->getLeaf() == true);
//                              REQUIRE(c1->getActive() == true);
//                      }
//                      THEN("The criterias vector size grows") {
//                              auto str1 = "criteria c1";
//                              auto c1 = ahpg->hierarchy->addCriteria(str1);
//                              REQUIRE(ahpg->hierarchy->getCriteriasSize() == size + 1);
//                      }
//                      THEN("Two criteiras with same name cannot be instantied") {
//                              auto str1 = "criteria c1";
//                              auto c1 = ahpg->hierarchy->addCriteria(str1);
//                              auto c2 = ahpg->hierarchy->addCriteria(str1);
//                              REQUIRE(std::string(c1->getName()) == str1);
//                              REQUIRE(c1 != c2);
//                              REQUIRE(c2 == NULL);
//                      }
//                      THEN("Create two criterias and set and edge") {
//                              auto str1 = "criteria c1";
//                              auto str2 = "criteria c2";
//                              auto c1 = ahpg->hierarchy->addCriteria(str1);
//                              auto c2 = ahpg->hierarchy->addCriteria(str2);
//                              REQUIRE(c1 != c2);
//                              REQUIRE(std::string(c1->getName()) == str1);
//                              REQUIRE(std::string(c2->getName()) == str2);
//                              int edgesSize = c1->getSize();
//                              ahpg->hierarchy->addEdge(c1, c2);
//                              REQUIRE(c1->getSize() == edgesSize + 1);
//                      }
//                      THEN("Create a Focus and Criteria and set an edge") {
//                              auto f = ahpg->hierarchy->addFocus("focus");
//                              auto c = ahpg->hierarchy->addCriteria("criteria");
//                              int edgesSize = f->getSize();
//                              ahpg->hierarchy->addEdge(f, c);
//                              REQUIRE(f->getSize() == edgesSize + 1);
//                      }
//              }
//              WHEN("The Alternative are added") {
//                      int size = ahpg->hierarchy->getAlternativesSize();
//                      auto alt = ahpg->hierarchy->addAlternative();
//                      THEN("The Alternative are set with default values") {
//                              auto re = alt->getResource();
//                              auto defRe = ahpg->hierarchy->getResource();
//                              REQUIRE(re != NULL);
//                              REQUIRE(defRe != NULL);
//                              int size_data = re->getDataSize();
//                              for(int i=0; i<size_data; i++) {
//                                      REQUIRE(re->getResource(i) == defRe->getResource(i));
//                                      REQUIRE(re->getResourceName(i) == defRe->getResourceName(i));
//                              }
//                      }
//                      THEN("The alternatives vector size grows") {
//                              REQUIRE(ahpg->hierarchy->getAlternativesSize() == size + 1);
//                      }
//              }
//              WHEN("The Resources changes") {
//                      auto def = *ahpg->hierarchy->getResource();
//                      THEN("New resource are added") {
//                              REQUIRE(ahpg->hierarchy->getResource()->getDataSize() == def.getDataSize());
//                              ahpg->hierarchy->addResource((char*)"a");
//                              REQUIRE(ahpg->hierarchy->getResource()->getDataSize() == def.getDataSize() + 1);
//                      }
//              }
//              WHEN("The Hierarchy are constructed") {
//                      Node* o = ahpg->hierarchy->addFocus("Qual host escolher?");
//                      Node* c = ahpg->hierarchy->addCriteria("largura de banda");
//                      Node* c2 = NULL;
//                      REQUIRE( o != NULL);
//                      REQUIRE( c != NULL);
//                      REQUIRE( strcmp(c->getName(), "largura de banda") == 0);
//                      c->setLeaf(true);
//                      ahpg->hierarchy->addSheets(c);
//                      float weight[4] = {1, 1, 1, 1};
//                      ahpg->hierarchy->addEdge(o, c, weight, 4);
//                      c = ahpg->hierarchy->addCriteria("vCPU");
//                      c->setLeaf(true);
//                      ahpg->hierarchy->addSheets(c);
//                      ahpg->hierarchy->addEdge(o, c, weight, 4);
//                      c = ahpg->hierarchy->addCriteria("QoS");
//                      c->setLeaf(true);
//                      ahpg->hierarchy->addSheets(c);
//                      ahpg->hierarchy->addEdge(o, c, weight, 4);
//                      c = ahpg->hierarchy->addCriteria("Rede");
//                      c->setLeaf(false);
//                      ahpg->hierarchy->addEdge(o, c, weight, 4);
//                      c2 = ahpg->hierarchy->addCriteria("L1");
//                      c2->setLeaf(true);
//                      ahpg->hierarchy->addSheets(c2);
//                      ahpg->hierarchy->addEdge(c, c2, weight, 4);
//                      c2 = ahpg->hierarchy->addCriteria("L2");
//                      c2->setLeaf(true);
//                      ahpg->hierarchy->addSheets(c2);
//                      ahpg->hierarchy->addEdge(c, c2, weight, 4);
//                      c2 = ahpg->hierarchy->addCriteria("L3");
//                      c2->setLeaf(true);
//                      ahpg->hierarchy->addSheets(c2);
//                      ahpg->hierarchy->addEdge(c, c2, weight, 4);
//                      c2 = ahpg->hierarchy->addCriteria("L4");
//                      c2->setLeaf(true);
//                      ahpg->hierarchy->addSheets(c2);
//                      ahpg->hierarchy->addEdge(c, c2, weight, 4);
//                      Node* a = ahpg->hierarchy->addAlternative();
//                      a->setName("a1");
//                      a = ahpg->hierarchy->addAlternative();
//                      a->setName("a2");
//                      a = ahpg->hierarchy->addAlternative();
//                      a->setName( "a3");
//                      a = ahpg->hierarchy->addAlternative();
//                      a->setName( "a4");
//                      ahpg->hierarchy->addEdgeSheetsAlternatives();
//                      float* weights = (float*) malloc (sizeof(float) * ahpg->hierarchy->getAlternativesSize());
//                      for(int i=0; i<ahpg->hierarchy->getAlternativesSize(); i++) {
//                              weights[i]= 0.5;
//                      }
//                      Node** sheets = ahpg->hierarchy->getSheets();
//                      for( int i=0; i<ahpg->hierarchy->getSheetsSize(); i++) {
//                              Edge** ed = sheets[i]->getEdges();
//                              for (int j = 0; j<sheets[i]->getSize(); j++) {
//                                      ed[j]->setWeights(weights, ahpg->hierarchy->getAlternativesSize());
//                              }
//                      }
//                      THEN("The hierarchy focus aren't NULL") {
//                              REQUIRE(ahpg->hierarchy->getFocus() != NULL);
//                      }
//                      THEN("The hierarchy criterias size aren't 0, the size corresponds to all "
//                           "criterias (included sheets criterias)") {
//                              REQUIRE(ahpg->hierarchy->getCriteriasSize() == 8);
//                      }
//                      THEN("The hierarchy sheets size aren't 0, his size corresponds to "
//                           "criterias that have edges with alternatives") {
//                              REQUIRE(ahpg->hierarchy->getSheetsSize() == 7);
//                      }
//                      THEN("The hierarchy alternatives size aren't 0") {
//                              REQUIRE(ahpg->hierarchy->getAlternativesSize() == 4);
//                      }
//                      WHEN("Execute the ahp synthesis") {
//                              ahpg->synthesisG();
//                              THEN(
//                                      "The hierarchy matrix, normalized matrix, pml and pg aren't NULL") {
//                                      auto f = ahpg->hierarchy->getFocus();
//                                      REQUIRE(f->getMatrix() == NULL);
//                                      REQUIRE(f->getNormalizedMatrix() == NULL);
//                                      // REQUIRE(f->getPml() == NULL);
//                                      REQUIRE(f->getPg() != NULL);
//                              }
//                      }
//              }
//      }
// }

SCENARIO("Applying the AHP example") {
	GIVEN("A Leader example") {
		AHPG* ahpg = new AHPG();
		REQUIRE(ahpg->hierarchy == NULL);
		ahpg->setHierarchyG();    // BUILD THE HIERARCHY
		// Focus
		auto f = ahpg->hierarchy->addFocus("Choose the most suitable leader");
		// Adding criterias
		auto c = ahpg->hierarchy->addCriteria("Experience");
		c->setLeaf(true);
		float fExperience[4] = {1, 4, 3, 7};
		ahpg->hierarchy->addSheets(c);
		ahpg->hierarchy->addEdge(f, c, fExperience, 4);
		auto c2 = ahpg->hierarchy->addCriteria("Education");
		c2->setLeaf(true);
		float fEducation[4] = {1 / 4., 1, 1 / 3., 3};
		ahpg->hierarchy->addSheets(c2);
		ahpg->hierarchy->addEdge(f, c2, fEducation, 4);
		c = ahpg->hierarchy->addCriteria("Charisma");
		c->setLeaf(true);
		float fCharisma[4] = {1 / 3., 3, 1, 5};
		ahpg->hierarchy->addSheets(c);
		ahpg->hierarchy->addEdge(f, c, fCharisma, 4);
		c = ahpg->hierarchy->addCriteria("Age");
		c->setLeaf(true);
		float fAge[4] = {1 / 7., 1 / 3., 1 / 5., 1};
		ahpg->hierarchy->addSheets(c);
		ahpg->hierarchy->addEdge(f, c, fAge, 4);
		// Add Alternatives Resources
		// Add Alternatives
		auto a = ahpg->hierarchy->addAlternative();
		a->setName("tom");
		a = ahpg->hierarchy->addAlternative();
		a->setName("Dick");
		a = ahpg->hierarchy->addAlternative();
		a->setName("Harry");
		// adding edges through all sheets and alternatives
		ahpg->hierarchy->addEdgeSheetsAlternatives();
		// First Criteria to Alternatives
		float w11[3] = {1, 1 / 4., 4};
		float w12[3] = {4, 1, 9};
		float w13[3] = {1 / 4., 1 / 9., 1};
		float* experienceWeights[3] = {w11, w12, w13};
		// Second Criteira to Alternatives
		float w21[3] = {1, 3, 1 / 5.};
		float w22[3] = {1 / 3., 1, 1 / 7.};
		float w23[3] = {5, 7, 1};
		float* educationWeights[3] = {w21, w22, w23};
		// Third Criteira to Alternatives
		float w31[3] = {1, 5, 9};
		float w32[3] = {1 / 5., 1, 4};
		float w33[3] = {1 / 9., 1 / 4., 1};
		float* charismaWeights[3] = {w31, w32, w33};
		// Fourth Criteira to Alternatives
		float w41[3] = {1, 1 / 3., 5};
		float w42[3] = {3, 1, 9};
		float w43[3] = {1 / 5., 1 / 9., 1};
		float* ageWeights[3] = {w41, w42, w43};

		float** alternativesWeights[4] = {experienceWeights, educationWeights, charismaWeights, ageWeights};

		auto sheets = ahpg->hierarchy->getSheets();
		int aSize = ahpg->hierarchy->getAlternativesSize();
		for (int i = 0; i < ahpg->hierarchy->getSheetsSize(); i++) {
			auto edges = sheets[i]->getEdges();
			for (int j = 0; j < sheets[i]->getSize(); j++) {
				edges[j]->setWeights(alternativesWeights[i][j], sheets[i]->getSize());
			}
		}
		// run the synthesis and consistency ahp functions
		ahpg->synthesisG();
		//ahpg->consistency();
		THEN("check the local priority") {
			auto pml = ahpg->hierarchy->getFocus()->getPml();
			// ahpg->printPml(ahpg->hierarchy->getFocus());
			printf("%lf %lf %lf %lf\n",pml[0], pml[1], pml[2], pml[3]);
			REQUIRE(pml[0] + pml[1] + pml[2] + pml[3] == 1);
			exit(0);
		}
		THEN("check the global priority") {
			auto pg = ahpg->hierarchy->getFocus()->getPg();
			// ahpg->printPg(ahpg->hierarchy->getFocus());
			REQUIRE(pg[0] + pg[1] + pg[2] == 1);
			REQUIRE(pg[0] < pg[1]);
			REQUIRE(pg[2] < pg[0]);
		}
	}
	// GIVEN("A Car example") {
	//
	//      AHPG* ahpg = new AHPG();
	//      ahpg->setHierarchyG();
	//      // BUILD THE HIERARCHY
	//      // Focus
	//      auto f = ahpg->hierarchy->addFocus("Choose the best car for the Jones Family");
	//      // Adding criterias
	//      // printf("FUCK 2\n");
	//      auto cost = ahpg->hierarchy->addCriteria("Cost");
	//      float w1[4] = {1, 3, 7, 3};
	//      cost->setLeaf(false);
	//      // printf("FUCK 3\n");
	//      ahpg->hierarchy->addEdge(f, cost, w1,4);
	//      auto safety = ahpg->hierarchy->addCriteria("Safety");
	//      float w2[4] = {1 / 3., 1, 9., 1};
	//      // printf("FUCK 4\n");
	//      safety->setLeaf(true);
	//      ahpg->hierarchy->addSheets(safety);
	//      ahpg->hierarchy->addEdge(f, safety, w2,4);
	//      auto style = ahpg->hierarchy->addCriteria("Style");
	//      // printf("FUCK 5\n");
	//      float w3[4] = {1 / 7., 1 / 9., 1, 1 / 7.};
	//      style->setLeaf(true);
	//      ahpg->hierarchy->addSheets(style);
	//      ahpg->hierarchy->addEdge(f, style, w3,4);
	//      auto capacity = ahpg->hierarchy->addCriteria("Capacity");
	//      float w4[4] = {1 / 3., 1., 7, 1};
	//      capacity->setLeaf(false);
	//      ahpg->hierarchy->addEdge(f, capacity, w4,4);
	//      // add cost childs
	//      auto purchase = ahpg->hierarchy->addCriteria("Purchase Price");
	//      float w5[4] = {1, 2, 5, 3};
	//      purchase->setLeaf(true);
	//      ahpg->hierarchy->addSheets(purchase);
	//      ahpg->hierarchy->addEdge(cost, purchase, w5,4);
	//      auto fuelCosts = ahpg->hierarchy->addCriteria("Fuel Costs");
	//      float w6[4] = {1 / 2., 1., 2, 2};
	//      fuelCosts->setLeaf(true);
	//      ahpg->hierarchy->addSheets(fuelCosts);
	//      ahpg->hierarchy->addEdge(cost, fuelCosts, w6,4);
	//      auto maintenanceCosts = ahpg->hierarchy->addCriteria("Maintenance Costs");
	//      float w7[4] = {1 / 5., 1 / 2., 1, 1 / 2.};
	//      maintenanceCosts->setLeaf(true);
	//      ahpg->hierarchy->addSheets(maintenanceCosts);
	//      ahpg->hierarchy->addEdge(cost, maintenanceCosts, w7,4);
	//      auto resaleValue = ahpg->hierarchy->addCriteria("Resale Value");
	//      float w8[4] = {1 / 3., 1 / 2., 2, 1};
	//      resaleValue->setLeaf(true);
	//      ahpg->hierarchy->addSheets(resaleValue);
	//      ahpg->hierarchy->addEdge(cost, resaleValue, w8,4);
	//      // Adding capacity childs
	//      auto cargoCapacity = ahpg->hierarchy->addCriteria("Cargo Capacity");
	//      float w9[2] = {1, 1 / 5.};
	//      cargoCapacity->setLeaf(true);
	//      ahpg->hierarchy->addSheets(cargoCapacity);
	//      ahpg->hierarchy->addEdge(capacity, cargoCapacity, w9,2);
	//      auto passengerCapacity = ahpg->hierarchy->addCriteria("Passenger Capacity");
	//      float w10[2] = {5, 1};
	//      passengerCapacity->setLeaf(true);
	//      ahpg->hierarchy->addSheets(passengerCapacity);
	//      ahpg->hierarchy->addEdge(capacity, passengerCapacity, w10,2);
	//
	//      // Add Alternatives Resources
	//      // Add Alternatives
	//      auto a = ahpg->hierarchy->addAlternative();
	//      a->setName("Accord Sedan");
	//      a = ahpg->hierarchy->addAlternative();
	//      a->setName("Accord Hybrid");
	//      a = ahpg->hierarchy->addAlternative();
	//      a->setName("Pilot SUV");
	//      a = ahpg->hierarchy->addAlternative();
	//      a->setName("CR-V SUV");
	//      a = ahpg->hierarchy->addAlternative();
	//      a->setName("Element SUV");
	//      a = ahpg->hierarchy->addAlternative();
	//      a->setName("Odyssey Minivan");
	//
	//      // adding edges through all sheets and alternatives
	//      ahpg->hierarchy->addEdgeSheetsAlternatives();
	//      // Setting the alternatives priority to Purchase Price
	//      float pp1[6] = {1, 9, 9, 1, 1 / 2., 5};
	//      float pp2[6] = {1 / 9., 1, 1, 1 / 9., 1 / 9., 1 / 7.};
	//      float pp3[6] = {1 / 9., 1, 1, 1 / 9., 1 / 9., 1 / 7.};
	//      float pp4[6] = {1, 9, 9, 1, 1 / 2., 5};
	//      float pp5[6] = {2, 9, 9, 2, 1, 6};
	//      float pp6[6] = {1 / 5., 7, 7, 1 / 5., 1 / 6., 1};
	//      // Setting the alternatives priority to Safety
	//      float ps1[6] = {1, 1, 5, 7, 9, 1 / 3.};
	//      float ps2[6] = {1, 1, 5, 7, 9, 1 / 3.};
	//      float ps3[6] = {1 / 5., 1 / 5., 1, 2, 9, 1 / 8.};
	//      float ps4[6] = {1 / 7., 1 / 7., 1 / 2., 1, 2, 1 / 8.};
	//      float ps5[6] = {1 / 9., 1 / 9., 1 / 9., 1 / 2., 1, 1 / 9.};
	//      float ps6[6] = {3, 3, 8, 8, 9, 1};
	//      // Setting the alternatives priority to capacity
	//      float pc1[6] = {1, 1, 1 / 2., 1, 3, 1 / 2.};
	//      float pc2[6] = {1, 1, 1 / 2., 1, 3, 1 / 2.};
	//      float pc3[6] = {2, 2, 1, 2, 6, 1};
	//      float pc4[6] = {1, 1, 1 / 2., 1, 3, 1 / 2.};
	//      float pc5[6] = {1 / 3., 1 / 3., 1 / 6., 1 / 3., 1, 1 / 6.};
	//      float pc6[6] = {2, 2, 1, 2, 6, 1};
	//      // Setting the alternatives Priority to Fuel Costs
	//
	//      float pfc1[6] = {1, 1 / (1.13), 1.41, 1.15, 1.24, 1.19};
	//      float pfc2[6] = {1.13, 1, 1.59, 1.3, 1.4, 1.35};
	//      float pfc3[6] = {1 / (1.141), 1 / (1.159), 1, 1 / (1.23), 1 / (1.14), (1 / 1.18)};
	//      float pfc4[6] = {1 / (1.15), 1 / (1.3), 1.23, 1, 1.08, 1.04};
	//      float pfc5[6] = {1 / (1.24), 1 / (1.4), 1.14, 1 / (1.08), 1, 1 / (1.04)};
	//      float pfc6[6] = {1 / (1.19), 1 / (1.35), 1.18, 1 / (1.04), 1.04, 1};
	//      // Setting the alternatives Priority to Resale Value
	//      float psv1[6] = {1, 3, 4, 1 / 2., 2, 2};
	//      float psv2[6] = {1 / 3., 1, 2, 1 / 5., 1, 1};
	//      float psv3[6] = {1 / 4., 1 / 2., 1, 1 / 6., 1 / 2., 1 / 2.};
	//      float psv4[6] = {2, 5, 6, 1, 4, 4};
	//      float psv5[6] = {1 / 2., 1, 2, 1 / 4., 1, 1};
	//      float psv6[6] = {1 / 2., 1, 2, 1 / 4., 1, 1};
	//      // Setting the alternatives Priority to Maintenance Costs
	//      float pm1[6] = {1, 1.5, 4, 4, 4, 5};
	//      float pm2[6] = {1 / (1.5), 1, 4, 4, 4, 5};
	//      float pm3[6] = {1 / 4., 1 / 4., 1, 1, 1.2, 1};
	//      float pm4[6] = {1 / 4., 1 / 4., 1, 1, 1, 3};
	//      float pm5[6] = {1 / 4., 1 / 4., 1 / (1.2), 1, 1, 2};
	//      float pm6[6] = {1 / 5., 1 / 5., 1, 1 / 3., 1 / 2., 1};
	//      // Setting the alternatives Priority to Style
	//      float pst1[6] = {1, 1, 7, 4, 9, 5};
	//      float pst2[6] = {1, 1, 7, 4, 9, 5};
	//      float pst3[6] = {1 / 7., 1 / 7., 1, 1 / 6., 3, 1 / 3.};
	//      float pst4[6] = {1 / 4., 1 / 4., 6, 1, 7, 5};
	//      float pst5[6] = {1 / 9., 1 / 9., 1 / 3., 1 / 7., 1, 1 / 5.};
	//      float pst6[6] = {1 / 5., 1 / 5., 3, 1 / 5., 5, 1};
	//      // Setting the alternatives Priority to Cargo Capacity
	//      float pcg1[6] = {1, 1, 1 / 2., 1 / 2., 1 / 2., 1 / 3.};
	//      float pcg2[6] = {1, 1, 1 / 2., 1 / 2., 1 / 2., 1 / 3.};
	//      float pcg3[6] = {2, 2, 1, 1, 1, 1 / 2.};
	//      float pcg4[6] = {2, 2, 1, 1, 1, 1 / 2.};
	//      float pcg5[6] = {2, 2, 1, 1, 1, 1 / 2.};
	//      float pcg6[6] = {3, 3, 2, 2, 2, 1};
	//
	//      float* pp[6] = {pp1, pp2, pp3, pp4, pp5, pp6};
	//      float* ps[6] = {ps1, ps2, ps3, ps4, ps5, ps6};
	//      float* pc[6] = {pc1, pc2, pc3, pc4, pc5, pc6};
	//      float* pfc[6] = {pfc1, pfc2, pfc3, pfc4, pfc5, pfc6};
	//      float* psv[6] = {psv1, psv2, psv3, psv4, psv5, psv6};
	//      float* pm[6] = {pm1, pm2, pm3, pm4, pm5, pm6};
	//      float* pst[6] = {pst1, pst2, pst3, pst4, pst5, pst6};
	//      float* pcg[6] = {pcg1, pcg2, pcg3, pcg4, pcg5, pcg6};
	//
	//      //    float*** alternativesWeights = {pp, ps, pc, pfc, psv, pm, pst, pcg}; ->
	//      //    This'll cause error on the PG answer,  because the vector is in wrong
	//      //    order.
	//      // CAUTION!!! The order of the alternatives Weights are HIGHLY dependent on
	//      // the order of hierarchy sheets.
	//      float** alternativesWeights[8] = {ps, pst, pp, pfc, pm, psv, pcg, pc};
	//
	//      auto sheets = ahpg->hierarchy->getSheets();
	//      int aSize = ahpg->hierarchy->getAlternativesSize();
	//      for (int i = 0; i < ahpg->hierarchy->getSheetsSize(); i++) {
	//              auto edges = sheets[i]->getEdges();
	//              for (int j = 0; j < sheets[i]->getSize(); j++) {
	//                      edges[j]->setWeights(alternativesWeights[i][j], sheets[i]->getSize());
	//              }
	//      }
	//
	//      ahpg->synthesisG();
	//      // ahpg->consistency();
	//
	//      THEN("Check the local priority") {
	//              auto pml = ahpg->hierarchy->getFocus()->getPml();
	//
	//              REQUIRE(pml[0] + pml[1] + pml[2] + pml[3] == 1);
	//      }
	//      THEN("Check the global priority") {
	//              auto pg = ahpg->hierarchy->getFocus()->getPg();
	//              REQUIRE(pg[0] + pg[1] + pg[2] + pg[3] + pg[4] + pg[5] == 1);
	//              REQUIRE(pg[5] > pg[0]);
	//              REQUIRE(pg[0] > pg[3]);
	//              REQUIRE(pg[3] > pg[1]);
	//              REQUIRE(pg[1] > pg[4]);
	//              REQUIRE(pg[4] > pg[2]);
	//      }
	// }
}
