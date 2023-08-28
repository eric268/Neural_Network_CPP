#include "../Include/pch.h"
#include "../Include/Neurons.h"

Neurons::Neurons() : 
	activation  { 0.0 }, 
	deltaBias   { 0.0 }, 
	deltaError  { 0.0 }, 
	deltaOutput { 0.0 } 
{}