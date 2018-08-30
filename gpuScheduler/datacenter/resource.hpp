#ifndef _RESOURCE_NOT_INCLUDED_
#define _RESOURCE_NOT_INCLUDED_
typedef struct {
	std::map<std::string, int> mInt; ///< map to represent all the int variables.
	std::map<std::string, double> mWeight; ///< map to represent all the float and double variables.
	std::map<std::string, std::string> mString; ///< map to represent all the strings variables.
	std::map<std::string, bool> mBool;///< map to represent all bool variables.
	int mIntSize, mWeightSize, mStringSize, mBoolSize;///< variables to represent theis respective map size, used to reduce the function overloads call (removing the call of .size() in the map).
} Resource;

#endif
