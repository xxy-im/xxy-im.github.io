# 武士数独(五重数独) 舞蹈链解法


Course 6125021 Combinatorics Homework 1，武士数独

<!--more-->

> 数独求解，第一个想到的方法就是DFS回溯。但是简单回溯法在求解单个数独时效率还能接受，放在五重数独（武士数独）上可以就有点差强人意了。我要是用它来解武士数独的话也没必要为这道题写篇博客了 :cat:。 

-------------

# 开搞

在此之前又仔细学习了一遍[DancingLinks](https://xxy.im/dancinglinks/)。DLX算法解数独的关键在于将数独转化为精确覆盖问题，这一步在单个矩阵的情况下还是比较容易的，但在武士数独上就比较繁琐了。  

## 定义数据结构
```c++
const static int SAMURAI_EDGE	= 21;
const static int SAMURAI_MATRIX = 441;
const static int SAMURAI_ROWS	= 405;
const static int SUDOKU_EDGE	= 9;
const static int SUDOKU_MATRIX	= 81;
const static int COLUMN_SIZE	= 1692;

class DLNode
{
public:
  DLNode * Left;           // 左结点
  DLNode *Right;          // 右结点
  DLNode *Up;             // 上结点
  DLNode *Down;           // 下结点
  DLNode *Col;            // 所属列结点

  int row;                // 行号
  int nums;               // 该列存在的结点个数（当结点为列结点时有效，否则为-1）

  DLNode(DLNode *Col, int n, int s = -1):   
      Left(this), Right(this), Up(this), Down(this), 
      Col(Col), row(n), nums(s){ if (Col) Col->Add2Colume(this); };
  ~DLNode() {};

  void Add2Row(DLNode *node);            // 添加结点到该行末尾
  void Add2Colume(DLNode *node);         // 添加结点到该列尾

  void RemoveCol();                      // 移除该结点所在的列
  void RecoverCol();                     // 还原列
  void Remove();                         // 移除该结点关联的行和列
  };

class DancingLinks
{
public:
  DancingLinks(int s[SAMURAI_EDGE][SAMURAI_EDGE]);
  ~DancingLinks();

  DLNode *Head;
  std::vector<DLNode *> Cols;     // 列向量
  std::vector<DLNode *> Ans;      // 保存结果

  bool DLX();                     // DLX算法求解
  void ShowResult(int result[SAMURAI_MATRIX]);              // 输出结果
};
```

> **数独规则:**  
> 1. 每个格子只能填一个数字  
> 2. 每行每个数字只能填一遍(1-9)  
> 3. 每列每个数字只能填一遍(1-9)  
> 4. 每宫每个数字只能填一遍(1-9) 
   
## 武士数独精确覆盖问题
武士数独有五个数独组成，需要 $21 \times 21$ 大小的矩阵存储数据，即 $441$ 个元素。给五个数独编号
![点击放大](https://xxy.im/storage/images/sudoku1.png "武士数独编号")
## 约束定义（索引从0开始）
- **定义441列**  
第0列：表示位置(0, 0)填了一个数字  
第1列：表示位置(0, 1)填了一个数字  
. . . . . .  
第20列：表示位置(0, 20)填了一个数字  
第21列：表示位置(1, 0)填了一个数字  
. . . . . .  
第440列：表示位置(20, 20)填了一个数字  
> 位置$(X,Y)$,  $Col = X \times 21 + Y$

- **定义405列**（5个数独，总共45行）  
第441列：0号数独的第0行填了数字1  
第442列：0号数独的第0行填了数字2  
. . . . . .   
第449列：0号数独的第0行填了数字9  
第450列：0号数独的第1行填了数字1  
. . . . . .  
第845列：4号数独的第8行填了数字9  
> 第N列定义为 第$S$号数独$X$行填了数字$Y$，它们之间的关系为  
> $N = 441 + S \times 81 + X \times 9 + (Y-1)$  

- **定义405列**（5个数独，总共45列）  
第846列：0号数独的第0列填了数字1  
第847列：0号数独的第0列填了数字2  
. . . . . .   
第857列：0号数独的第0列填了数字9  
第858列：0号数独的第1列填了数字1  
. . . . . .  
第1250列：4号数独的第8列填了数字9  
> 第N列定义为 第$S$号数独$X$列填了数字$Y$，它们之间的关系为  
> $N = 441 + 405 + S \times 81 + X \times 9 + (Y-1)$  

- **定义441列**（$21\times21$矩阵，总共49个宫,为方便计算没有删去空白的宫）
第1251列：第0宫填了数字1  
第1252列：第0宫填了数字2  
. . . . . .  
第1259列：第0宫填了数字9  
第1260列：第1宫填了数字1  
. . . . . .  
第1691列：第48宫填了数字9
> 第N列定义为 第$S$宫填了数字$D$,它们之间的关系为  
> $N = 441 + 405 + 405 + S \times 9 + (D-1)$   

--------
*由上1692列完成了对武士数独的精确覆盖问题约束定义*

### 初始化Dancing Links
用上图数独为例，(0, 0) 位置为 9，转换为DancingLinksz中的一行，则第0，449, 857, 1259列为 1 (即存在结点)，其余列为 0。 

**Dancing Links初始化**
```c++
DancingLinks::DancingLinks(int sam[SAMURAI_EDGE][SAMURAI_EDGE])
{
	Head = new DLNode(nullptr, 0);

	// 创建列结点 1692个
	for (int i = 0; i < COLUMN_SIZE; i++)
	{
		auto t = new DLNode(nullptr, 0, 0);
		Head->Add2Row(t);
		Cols.push_back(t);
	}

	std::vector<DLNode *> Rows;     // 保存初始已存在数字的结点

	for (int r = 0; r < SAMURAI_EDGE; r++)
	{
		for (int c = 0; c < SAMURAI_EDGE; c++)
		{
			for (int d = 0; d < SUDOKU_EDGE; d++)
			{
				// 计算行数
				int row = (r * SAMURAI_EDGE * SUDOKU_EDGE) + (c * SUDOKU_EDGE) + d;

				int sq = (c / 3) + ((r / 3) * 7);

				int t = VALID_SQUARE[sq];

				if (t > 0)
				{
					auto node = new DLNode(Cols[r * SAMURAI_EDGE + c], row);

					for (int i = 0; i < 2; i++)
					{
						// 判断sq号宫属于第几号数独
						int sd = (t > 5 && !i) ? 4 : t-1;

						// 当前r，c属于sd号数独的几行几列
            //（感觉用数组索引更方便， 一开始我是直接硬算行列，后来在网上看到有人用数组的方式实现）
						int sdr = SUDOKU_ROW[sd][r];
						int sdc = SUDOKU_COLUMN[sd][c];

						// 五个数独 总共45行 1-9数字情况 405列
						node->Add2Row(new DLNode(Cols[SAMURAI_MATRIX +
							(sd * SUDOKU_MATRIX) +
							(sdr * SUDOKU_EDGE) + d], row));

						node->Add2Row(new DLNode(Cols[SAMURAI_MATRIX + SAMURAI_ROWS +
							(sd * SUDOKU_MATRIX) +
							(sdc * SUDOKU_EDGE) + d], row));

						if (t < 6) i++;
						t -= 5;
					}

					node->Add2Row(new DLNode(Cols[SAMURAI_MATRIX + SAMURAI_ROWS + SAMURAI_ROWS +
						(sq * SUDOKU_EDGE) + d], row));

					if (sam[r][c] == (d + 1))
					{
						Rows.push_back(node);
					}
				}
			}
		}
	}

	for (auto col = Head->Right; col != Head; col = col->Right)
	{
		if (!col->nums) col->RemoveCol();
	}

	for (auto iter = Rows.begin(); iter != Rows.end(); iter++)
	{
		(*iter)->Remove();
		Ans.push_back(*iter);
	}
}
```

**算法执行过程**
```c++
bool DancingLinks::DLX()
{
	if (Head->Right == Head)
	{
		auto result = new int[Ans.size()];

		for (int i = 0; i < Ans.size(); i++)
		{
			result[i] = Ans[i]->row;
		}
		ShowResult(result);

		return true;
	}

	DLNode *col = nullptr;
	int min = INT_MAX;

	// 找到列元素最少的列
	for (auto c = Head->Right; c != Head; c = c->Right)
	{
		if (min > c->nums)
		{
			col = c;
			min = c->nums;
		}
	}

	col->RemoveCol();

	for (auto node = col->Down; node != col; node = node->Down)
	{
		Ans.push_back(node);

		for (auto rnode = node->Right; rnode != node; rnode = rnode->Right)
		{
			rnode->Col->RemoveCol();
		}

		if (DLX()) return true;

		for (auto lnode = node->Left; lnode != node; lnode = lnode->Left)
		{
			lnode->Col->RecoverCol();
		}

		Ans.pop_back();
	}
	col->RecoverCol();

	return false;
}
```  

# 结果输出
**数独一：**  
![点击放大](https://xxy.im/storage/images/ss1.png "武士数独1")
  
**数独二：**  
![点击放大](https://xxy.im/storage/images/ss2.png "武士数独2")

# 完整代码
https://github.com/xxy-im/SudokuNinja （代码持续优化中）  

# 小结
其本质虽然还是DFS回溯，但是在Dancing Links这一数据结构的加持下，回溯效率大大提升，求解时间**小于0.1s**，在内存暴增的年代，用些许内存的占用去换取运行时间的加速还是划算的。  
  
后续计划在此基础上增加OCR功能，并优化代码使其适配任意形状数独 ~~~（没时间就算了）~~~。
# 参考文章
https://en.wikipedia.org/wiki/Dancing_Links  
https://www.cnblogs.com/grenet/p/3163550.html  
https://www.acwing.com/solution/acwing/content/3843/
