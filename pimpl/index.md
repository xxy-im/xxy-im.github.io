# [C++技法] PImpl


"Pointer to implementation", 指向实现的指针。将一个类的实现细节从其对象中移除，也是一种解耦方法。

<!--more-->

# PImpl
> 使用私有的成员指针指向类的成员，是一种实现数据隐藏，最小化耦合和分离接口的现代C++编程技巧。

先看一段官方的PImpl代码
```c++
// interface (widget.h)
class widget
{
    // public members
private:
    struct impl;
    std::unique_ptr<impl> pImpl;
};
 
// implementation (widget.cpp)
struct widget::impl
{
    // implementation details
};
```  
可以看到```widget```类中使用了一个```unique```指针指向```impl```这个内部类。这样的好处主要有:
1. ABI(Application Binary Interface, 二进制接口) 稳定，即不会打破二进制兼容。
2. 降低编译依赖项，缩短编译时间。更改成员及实现时只需重新编译成员的源文件，而不需要重新编译所有使用了这个类的用户。
3. 接口与实现分离，提高接口的稳定性。
4. 降低耦合性。
5. 将实现隐藏，头文件变得整洁。

> 主要缺点是性能会受点影响，因为成员都是用指针间接访问的。

## std::unique_ptr
可以看到上面的代码使用的```std::unique_ptr```这个智能指针。这是C++11中基于RAII(Resource acquisition is initialization)思想引入的一个智能指针。例如，定义指针p ```std::unique_ptr<T> p = std::make_unique<T>()```，这时就不需要手动管理p指向的内存了，因为```std::unique_ptr```的析构函数会自动调用```delete p```。

需要注意的是 ```std::unique_ptr```是禁止拷贝的，所以```widget```也无法使用拷贝构造函数，但可以使用移动构造函数。


## 完善实现
因为类的定义中还有一个未实现的内部类，所以```widget```并不是一个完整的类，因此编译器不能为其自动生成构造和析构函数。此时需要在```widget.cpp```中显示的定义它的构造和析构函数，即使是使用```=default```也必须放在```cpp```中。  

PImpl的完整代码：
引用自:[en.cppreference.com/w/cpp/language/pimpl](https://en.cppreference.com/w/cpp/language/pimpl)
```c++
// interface (widget.hpp)
#include <iostream>
#include <memory>

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
 
// ---------------------------
// implementation (widget.cpp)
// #include "widget.hpp"
 
class widget::impl
{
    int n; // private data
public: 
    impl(int n) : n(n) {}
};
 
void widget::draw() const { pImpl->draw(*this); }
void widget::draw() { pImpl->draw(*this); }
widget::widget() = default;
widget::widget(int n) : pImpl{std::make_unique<impl>(n)} {}
widget::widget(widget&&) = default;
widget::~widget() = default;
widget& widget::operator=(widget&&) = default;
```

## 其他
一般来说，工厂模式也能消除接口实现的编译时依赖，但工厂模式不是ABI稳定的，因为需要修改虚函数表。  

PImpl类是对移动友好的；把大型的类重构为可以移动的PImpl，可以提升容器进行操作的算法性能，但也具有额外的运行时开销，因为任何在被移动对象上允许使用并需要访问私有实现的公开成员函数都必须进行空指针检查。
