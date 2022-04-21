# [C++技法] 多线程编程



<!--more-->

## 进程与线程
这个没啥好讲的吧，但凡稍微学了点操作系统或者复习了408的应该都知道了。  
简单说下它们之间的关系：  
- 线程从属于进程，一个进程可以拥有多个线程
- 每个线程除了独立拥有很小的一点栈外，共享进程的内存空间    

### Why多线程
用我们最常用的浏览器来举例，通常我们都会在浏览器上很多标签页，有的页面听歌，有的用来搜索，还有用来下载文件。浏览器不可能让你听完歌再继续后面的操作吧，这时候就需要多线程了。一个线程处理听歌，一个线程处理下载，等等等等。实现单进程多任务场景。  
在任务管理器中可以看到Chrome开了这么多线程。(明明就开了几个网页，给我开了这么多个线程，咱也不懂)
![点击放大](https://cdn.jsdelivr.net/gh/xxy-im/storage@gh-pages/images/chrome-multitread.png "Chrome浏览器进程")

## 现代C++的多线程
C++11之前，需要多线程编程的话需要使用`pthread`库，C++11开始引入了`std::thread`实现多线程。但这玩意儿其实还是用`pthread`实现的，所以用`g++`编译的话还得加`-lpthread`...  
```c++
// std::thread 构造函数
thread() noexcept;                                  // 1
thread( thread&& other ) noexcept;                  // 2
template< class Function, class... Args >          
explicit thread( Function&& f, Args&&... args );    // 3
thread( const thread& ) = delete;
```
主要用第三个比较多，参数可以用`lambda`表达式。貌似只要是`Callable`的就行，官方示例里有个定义的`operator()`的类也可以成功创建线程。

**简单实现上面浏览器的场景**
```c++
#include <iostream>
#include <thread>
#include <string>

// 功能函数
void BrowsePage()
{
	for (int i = 0; i < 10; i++)
	{
		std::cout << "Searching something on Google..... " << std::endl;
		std::this_thread::sleep_for(std::chrono::seconds(1));               // 新标准的睡眠函数，不再需要用sleep这种了
	}
}
void ListenMusic()
{
	for (int i = 0; i < 10; i++)
	{
		std::cout << "正在播放《以父之名》..... " << std::endl;  
		std::this_thread::sleep_for(std::chrono::seconds(2));
	}
}
void Download(std::string filename)
{
	for (int i = 0; i < 20; i++)
	{
		std::cout << "Downloading " << filename 
			<< " ( " << i * 10 << "% )......" << std::endl;
		std::this_thread::sleep_for(std::chrono::seconds(1));
	}
	std::cout << filename << "Download completed" << std::endl;
}

int main()
{
    std::thread t1(ListenMusic);
	std::thread t2([&]() {
		std::string filename;
		std::cin >> filename;
		Download(filename);
		});
	BrowsePage();
	t1.join();              // 相当于一个wait，等待这个线程结束了再继续执行后面的语句
	t2.join();
           
	return 0;
}
```
运行后可以边浏览网页边输入需要下载的文件然后回车开始下载。如果没加`join`的话就会有可能导致程序已经结束了但`t1`,`t2`线程还没结束但也会随着进程结束而强行终止执行。  
**注意** `BrowsePage()`这里我没用线程来执行，所以必须放在最后面，如果放在`t1`或者`t2`线程前面的话那么就会导致线程会等待`BrowsePage`执行完才会开始执行。

> `std::thread`也是遵循RAII思想的，所以当线程所在函数执行结束时会自动调用`std::thread`的析构函数。但是可以使用`detach()`将线程分离，使其不再由当前对象管理，而是在线程运行结束后自动销毁。不过这东西也不怎么好，就不介绍了

### 线程池
最low的线程池可以用`std::vector`实现
```c++
std::vector<std::thread> tpool;
void DoIt()
{
	std::thread t1(BrowsePage);
	std::thread t2(ListenMusic);
	std::thread t3([&]() {
		std::string filename;
		std::cin >> filename;
		Download(filename);
		});

	tpool.push_back(std::move(t1));		// std::thread 禁止拷贝
	tpool.push_back(std::move(t2));
	tpool.push_back(std::move(t3));
}
int main()
{
	DoIt();
	for (auto& t : tpool)
	{
		t.join();
	}
	
	return 0;
}
```
但是这样手动把每个线程`join`实在太low了。  
可以自定义一个管理线程池的类，在类的析构函数中加入线程的`join`即可。(其实就是把for循环换了个地方那它自动会执行，看起来代码整洁点而已)

> C++20 中的`std::jthread`会在析构时候自动join

## 同步与互斥
就像进程需要处理临界区互斥一样，多线程中因为共享着进程的内存空间，所以也需要有互斥手段。当然同步也是一样的。
借用多进程互斥里经典的存钱取钱的场景(虽然现实中不可能在一个进程里存钱又取钱，但道理都相通嘛)

```c++
// 假设当前余额为0，现在在ATM机上先存1000个w，在取500个w
// 别在意故事细节
#include <iostream>
#include <thread>

int money = 0;		// 当前余额

void deposit(int m)
{
	int cur = money;			// 得先把在的余额取出来吧
	m = m > 10000 ? 10000 : m;	// ATM机一次最多一个w
	for (volatile int i = 0; i < 100; i++);		// 假装有个后台处理过程
	cur += m;

	money = cur;				// 然后新的余额
}

void withdraw(int m)
{
	int cur = money;
	m = m > 10000 ? 10000 : m;	// ATM机一次最多一个w
	for (volatile int i = 0; i < 100; i++);
	cur -= m;

	money = cur;
}

int main()
{
	std::thread t1([&] {
		for (int i = 0; i < 1000; i++) deposit(10000);
		});
	std::thread t2([&] {
		for (int i = 0; i < 500; i++) withdraw(10000);
		});
	t1.join();
	t2.join();

	std::cout << money << std::endl;
	return 0;
}
```
多次运行后会发现结果都不同。有时钱多有时钱少了。这时就需要用到互斥手段以保证程序运行的正确性了。  

### std::mutex
看名字也知道是个互斥锁，用来给资源上锁的。修改两个功能函数。  
```c++
#include <mutex>		// 引入头文件
std::mutex mtx;			// 定义一个互斥锁

void deposit(int m)
{
	mtx.lock();					// 上锁
	int cur = money;			// 得先把在的余额取出来吧
	m = m > 10000 ? 10000 : m;	// ATM机一次最多一个w
	for (volatile int i = 0; i < 100; i++);		// 假装有个后台处理过程
	cur += m;

	money = cur;				// 然后新的余额
	mtx.unlock();				// 用完money解锁
}

void withdraw(int m)
{
	mtx.lock();
	int cur = money;
	m = m > 10000 ? 10000 : m;	// ATM机一次最多一个w
	for (volatile int i = 0; i < 100; i++);
	cur -= m;

	money = cur;
	mtx.unlock();
}
```
> `mtx.lock()`如果加锁失败会一直尝试上锁直到成功为止，还可以使用`mtx.try_lock()`仅尝试一次上锁，成功返回`true`，失败返回`false`。

## 死锁
有互斥当然就会有死锁。
- `std::lock`: 一次性执行多个互斥锁的`lock()`，也可以作为一种防止死锁的手段
- `std::recursive_mutex`: 递归互斥锁，当同一个线程对一个资源多次上锁也会造成死锁，如果一定要写这样的代码的话可以用这个互斥锁防止死锁。  

## 读写者问题
针对读写者问题，C++14开始引入了一个专门的读写锁`std::shared_mutex`，比用其他互斥锁实现读写者问题性能提升了很多。  
`std::shared_mutex`有两对加锁解锁方式:  
- `lock()`和`unlock()`: 互斥性的，用于写者线程
- `lock_shared()`和`unlock_shared()`: 共享性的，lock后其他线程也可以访问，可以用于读者线程
> 除了用`std::shared_mutex`还可以用`std::shared_lock<>`模板函数把其他的互斥锁用shared方式上锁。

## 条件变量
`std::condition_variable`，类似多进程里的信号量

### 生产者消费者
用条件变量实现生产者消费者问题
```c++
#include <iostream>
#include <vector>
#include <random>
#include <thread>
#include <mutex>
#include <condition_variable>

int products[100];		// 只能存放100个商品
int count = 0;			// 当前商品容量

std::mutex mtx;
std::condition_variable cv;

std::default_random_engine random;
std::uniform_int_distribution<int> dis(0, 1000);
// 生产者线程
void producer()
{
	std::unique_lock<std::mutex> lck(mtx);						// =执行mtx.lock()
	cv.wait(lck, [&]() {return count < 100; });
	std::this_thread::sleep_for(std::chrono::milliseconds(100)); // 生产耗时100ms
	products[count++] = dis(random);	// 随机生产个0-1000均匀分布的商品编号
	lck.unlock();
	cv.notify_one();	// 唤醒一个条件变量
	std::cout << "生产了一件商品" << products[count-1] << "，当前商品数为 " << count << std::endl;
}

void consumer()
{
	std::unique_lock<std::mutex> lck(mtx);		// =执行mtx.lock()
	cv.wait(lck, [&]() {return count > 0; });
	count--;
	lck.unlock();
	cv.notify_one();
	std::cout << "卖出了一件商品，当前商品数为 " << count << std::endl;
}

int main()
{
	// 一个生产者，两个消费者
	std::thread t1([&]() {
		while (1)
		{
			producer();
		}
		});
	std::thread t2([&]() {
		while (1)
		{
			std::this_thread::sleep_for(std::chrono::milliseconds(200)); // 线程2卖东西要200ms 
			consumer();
		}
		});
	std::thread t3([&]() {
		while (1)
		{
			std::this_thread::sleep_for(std::chrono::milliseconds(300)); // 线程3卖东西要300ms 
			consumer();
		}
		});

	t1.join();
	t2.join();
	t3.join();

	return 0;
}
```

## 其他
- `std::async`: 也可用于创建线程，自身返回一个`std::future`。可以捕获返回值，同时可以配合`wait()`和`wait_for`使用。加上`std::launch::deferred`参数可以是其更加灵活。  
- `std::promise`: `std::async`其实就是用这东西实现的，你可以用这个自己手动实现`std::async`，C++就是这么贴心，帮你做了还告诉你怎么做的 :dog:。
- `std::lock_guard`: 一个符合RAII的`std::mutex`，构造时会自动执行`lock()`，析构时自动`unlock()`。
- `std::unique_lock`: **(一般推荐用这个)** 更灵活的`std::lock_guard`，因为`std::lock_guard`严格的在析构时才会`unlock()`，而有时需要提前`unlock()`就可以用这个。  
  还可以用`std::try_to_lock`参数实现`std::mutex`的`try_lock()`的效果。还可以用`std::mutex`来初始化`std::unique_lock`  
  还可以在构造时使用`std::defer_lock`来推迟执行`lock()`。(反正就奇奇怪怪的需求C++都能满足你。  
- `std::timed_mutex`: 一个可以设置等待时间的互斥锁，`try_lock_for()`函数中用`std::chrono`设定时间，还可以使用`try_lock_until()`
- `std::scoped_lock`: RAII版本的`std::lock`。
- `std::recursive_timed_mutex`: 带time版本的`std::recursive_mutex`。
- `std::atomic`: 原子类型对象，锁住内存总线，让CPU不去进行乱序执行优化策略。所以从不同线程访问原子类型对象不会导致数据竞争(data race)。
