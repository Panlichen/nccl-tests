#include <iostream>
#include <vector>
#include <algorithm>

using namespace std;

const int LEN = 8;

vector<vector<int>> ans;

bool row_valid(int num_valid_rows, vector<int> &row) {
  if (num_valid_rows == 0) {
    return true;
  }
  for (int i = 0; i < num_valid_rows; ++i) {
    for (int j = 0; j < LEN; ++j) {
      if (row[j] == ans[i][j]) {
        return false;
      }
    }
  }
  return true;
}

int main () {
  vector<int> row = {0, 5, 2, 6, 4, 1, 3, 7};
  while (ans.size() < LEN) {
    next_permutation(row.begin(), row.end());
    if (row_valid(ans.size(), row)) {
      ans.emplace_back(row);
    }
  }

  for (int i = 0; i < ans.size(); ++i) {
    cout << "{";
    for (int j = 0; j < LEN; ++j) {
      cout << ans[i][j];
      if (j < LEN - 1) {
        cout << ", ";
      }
    }
    cout << "}, \n";
  }

  return 0;
}