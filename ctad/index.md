# [C++技法] CTAD (since C++17)


C++17开始引入的类模板实参推导

<!--more-->

# 使用类模板

## 远古C++
在远古C++中实例化一个类模板往往需要将参数类型全部写出来  
比如使用STL中的`std::pair`
```c++
std::pair<int, double> p = std::make_pair<int, double>(2, 1.5);
```
可以看到代码非常冗余，如果遇到更长的类名简直看不下去。

## C++11
然后到了C++11引入了`auto`自动推断类型
```c++
auto p = std::make_pair<int, double>(2, 1.5);
```
代码一下缩短了很多，这里`<int, double>`其实也是不用写的。

## C++17 (CTAD)
编译器自动从类模板初始化值的类型推导出模板实参类型
```c++
std::pair p(2, 4.5)         // 推导出std::pair<int, double>
std::tuple t(4, 3, 2.5)     // 等价于 auto t = std::make_tuple(4, 3, 2.5)
```

**new表达式同样适用：**  
```c++
template<class T> struct A
{
    A(T){};
};
auto y = new A{1};      // 等价于 new A<int>(1);
```

> 同时支持聚合推导，因为我用的少就不多介绍了，记得初始化时用`{}`，而不是`()`就行了。

### 用户定义的推导指引
除了靠编译器自动进行参数类型推导外，还可以自定义推导，语法结构类型声明`lambda`函数的返回类型
```c++
// explicit(可选) 模板名 (形参声明子句) -> 简单模板标示;
template<class T> struct A
{
	A(T) {};
	template<class T> A(T a, T b) {};
};

// 自定义推导，我这里让它变成了两个参数的构造函数全部推导为A<int>
template<class T>
A(T, T)->A<int>;

std::string str = "hello";
auto t = A(str, str);       // 等价于 A<int>{};
```

## 其他
需要注意的是，CTAD只有在**目标类型的构造函数使用模板参数的情况**下才能进行推导。当然还有将某个类禁止CTAD的方法就不一一概述了。
