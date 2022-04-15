# [C++技法] 右值引用与移动语义


了解右值和右值引用的概念以及移动语义的实现。

<!--more-->

> 右值引用(rvalue reference)，是C++11标准提出的一类数据类型。  
> 可用于实现移动语义(move semantic)与完美转发(perfect forwarding)。

## 右值
何为右值(r-value)，说人话就是**只能**放在等号右边的东西。例如`int a = 1`这个表达式中，`a`在等号左边，所以`a`是左值，而`1`是右值。  
右值通常为一个表达式，是赋值计算产生临时生成的中间变量。
## 右值引用
C++中，通常的引用是指左值引用，用符号`&`表示，而右值引用符号为`&&`。
```c++
int a = 1;
int& ref = a;       // 左值引用
```
在上述代码中，定义了一个对`a`的左值引用，但是`&`符号不能对`1`引用，`int &ref = 1`的非法的。  
但是可以使用`int &&ref = 1`，定义一个对`1`的右值引用。  
```c++
int a = 1;
int& ref = a;       // 左值引用

// int& ref = 1;    // error

int&& rref = 1;     // 右值引用
// int&& r_ref = a;  // error，右值引用不可指向左值
r_ref = 2;           // 右值引用也可以修改值
```
> 可以看出来这里的右值引用自身是一个左值（有名字的右值引用自身是左值）。
### std::move
`std::move`一般理解为移动操作，在[PImpl](https://xxy.im/pimpl/)讲过的`std::unique_ptr`这个智能指针是禁止拷贝的，这是便可使用`std::move`对其进行移动操作。但`std::move`的原理是将左值转化为右值，底层操作中并没有实现内存的移动啥的。（如果没理解的话这就是个坑）
```c++
int a = 1;
int& ref = a;
int&& r_ref = std::move(a);     // 将a转化为左值 与 int&& rref = 1 等价
r_ref = 2;          // 等价 a = 2
```
但是和`int&& rref = 1`不同的是，此时`r_ref`也相当于`a`的一个左值引用。同时可以看出`std::move`根本没把`a`给移掉，因为像`int`这样的基本类型`std::move`对其是没有影响的。像`string`、`std::unique_ptr`这样的`move`就会变空了。*要养成移动后不在使用的习惯*

### 右值引用作函数参数  
```c++
void func(int &&v)
{
    // do something
}

int a = 1;
func(std::move(a));     // ok
func(2);                // ok
```
单从性能上来看，左右值引用都避免了传参拷贝。
顺带提一下，C++规定 `&&` 可以自动转化为`const&`，所以当形参为`void func(int const& v)`时调用`func(2)`其实是隐含了一个转换。但右值引用比`const`引用更灵活，因为它还是可以修改的。

## 移动语义
### 移动构造函数
在[PImpl](https://xxy.im/pimpl/)中也可以看到`widget`类中移动构造函数的参数为右值引用。
```c++
class widget
{
    class impl;
    std::unique_ptr<impl> pImpl;
public:
    widget();
    explicit widget(int);
    ~widget();
    widget(widget&&);   // 移动构造
    widget(const widget&) = delete;
    widget& operator=(widget&&);    // 移动赋值
    widget& operator=(const widget&) = delete;
};

int main()
{
    widget w;
    widget wm = widget(std::move(w));
    // do something
}
```
这样做的好处同样是比用`const`引用更加灵活，可以做浅拷贝提升性能。

### 容器避免深拷贝
STL类大都支持移动语义函数，比如`vector`就可以用`std::move`避免深拷贝以提升性能
```c++
std::vector<std::string> sVec;
std::string str = "hello";
sVec.push_back(std::move(str));     // 避免是对str的拷贝，性能得到提升
```
> 可移动对象在*需要拷贝且被拷贝者之后不再被需要*的场景，可以使用`std::move`触发移动语义，提升性能。

## 其他
### std::forward
```std::forward```叫做**完美转发**，和`std::move`一样，这货跟转发没半毛钱关系。也是用于类型转换。  
它不仅可以把左值转为右值，还可以反过来把右值转为左值。  
使用方法：
```c++
std::forward<T>(v);
// 1. 当T为左值引用时，v被转换为T类型的左值引用
// 2. 否则，v转换为T类型的右值引用  
```
这东西使用场景不多，我也不太懂，就不多做介绍了。   

更多右值引用技巧可看这个 https://zhuanlan.zhihu.com/p/107445960
