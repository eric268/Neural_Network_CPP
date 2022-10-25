#include <iostream>
#include <string>
#include <vector>
#include <list>
#include <algorithm>
#include <unordered_map>
#include <unordered_set>
#include <concepts>
#include <limits>
#include <chrono>
#include <fstream>
#include <sstream>
#include <deque>
#include <stack>
#include <queue>

//Windows functions
#include <windows.h>
#include <tchar.h>
#include <stdlib.h>
#include <string.h>

#define Print( s )		         \
{								 \
std::wstring text = s;			 \
OutputDebugString(text.c_str()); \
}
