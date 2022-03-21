# Git 指令

```git
git init					//初始化git本地仓库
git add file.txt   		  	//添加文件
git status       	 	  	//查看暂存区
git commit -m  '提交日志'  	 //提交文件 -m附带日志
git reset file.txt		 	//将add之后的文件撤销
git log   				 	//查看记录
git reset --hard HEAD~1   	//版本回退一个版本
git reset --hard 标识符 		//版本重置到标识符所在的版本
git reflog 				    //查看指针HEAD所作操作
git checkout -- file.txt	//从仓库恢复文件
git ls-files				//查看仓库内文件
git rm file.txt				//从仓库删除文件，不可checkout

git clone website.git		//从网站克隆代码
git remote add origin website.git  //绑定仓库 名字为origin
git push -u origin master	//首次推送 将origin推送至master

//使用SSH进行推送

ssh-keygen -t rsa -C '邮箱地址'  //本地生成SSH公钥与私钥
ssh -T git@github.com			//进行匹配

//新仓库创建之后配置示例：
git init
git add README.md
git commit -m "first commit"
git branch -M main
git remote add origin git@github.com:Shawyangg/NOTE.git
git push -u origin main


```

