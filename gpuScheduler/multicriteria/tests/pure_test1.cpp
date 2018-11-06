#include "../../thirdparty/catch.hpp"
#include "../ahp.hpp"
#include <random>

int main(){
	AHP *ahp = new AHP();
	ahp->setHierarchy(); // BUILD THE HIERARCHY
// Focus
	auto f = ahp->hierarchy->addFocus("Choose the most suitable leader");
// Adding criterias
	auto c = ahp->hierarchy->addCriteria("Experience");
	c->setLeaf(true);
	float fExperience[4] = {1, 4, 3, 7};
	ahp->hierarchy->addSheets(c);
	ahp->hierarchy->addEdge(f, c, fExperience, 4);
	auto c2 = ahp->hierarchy->addCriteria("Education");
	c2->setLeaf(true);
	float fEducation[4] = {1 / 4., 1, 1 / 3., 3};
	ahp->hierarchy->addSheets(c2);
	ahp->hierarchy->addEdge(f, c2, fEducation, 4);
	c = ahp->hierarchy->addCriteria("Charisma");
	c->setLeaf(true);
	float fCharisma[4] = {1 / 3., 3, 1, 5};
	ahp->hierarchy->addSheets(c);
	ahp->hierarchy->addEdge(f, c, fCharisma, 4);
	c = ahp->hierarchy->addCriteria("Age");
	c->setLeaf(true);
	float fAge[4] = {1 / 7., 1 / 3., 1 / 5., 1};
	ahp->hierarchy->addSheets(c);
	ahp->hierarchy->addEdge(f, c, fAge, 4);
// Add Alternatives Resources
// Add Alternatives
	auto a = ahp->hierarchy->addAlternative();
	a->setName("tom");
	a = ahp->hierarchy->addAlternative();
	a->setName("Dick");
	a = ahp->hierarchy->addAlternative();
	a->setName("Harry");
// adding edges through all sheets and alternatives
	ahp->hierarchy->addEdgeSheetsAlternatives();
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

	auto sheets = ahp->hierarchy->getSheets();
	int aSize = ahp->hierarchy->getAlternativesSize();
	for (int i = 0; i < ahp->hierarchy->getSheetsSize(); i++) {
		auto edges = sheets[i]->getEdges();
		for (int j = 0; j < sheets[i]->getSize(); j++) {
			edges[j]->setWeights(alternativesWeights[i][j], sheets[i]->getSize());
		}
	}
// run the synthesis and consistency ahp functions
	ahp->synthesis();
//ahp->consistency();
	auto pml = ahp->hierarchy->getFocus()->getPml();
	// ahp->printPml(ahp->hierarchy->getFocus());
	auto pg = ahp->hierarchy->getFocus()->getPg();
	ahp->printPg(ahp->hierarchy->getFocus());
	delete(ahp);
}
