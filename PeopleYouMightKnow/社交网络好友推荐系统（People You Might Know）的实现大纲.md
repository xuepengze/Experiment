# **社交网络好友推荐系统（People You Might Know）的实现**

## 1.**编写 MapReduce 代码**

### a.**FriendCountWritable （用于存储每个好友推荐对的信息）**

```java
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.Writable;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;

public class FriendCountWritable implements Writable {
    public Long user;
    public Long mutualFriend;

    public FriendCountWritable(Long user, Long mutualFriend) {
        this.user = user;
        this.mutualFriend = mutualFriend;
    }

    public FriendCountWritable() {
        this(-1L, -1L);
    }

    @Override
    public void write(DataOutput out) throws IOException {
        out.writeLong(user);
        out.writeLong(mutualFriend);
    }

    @Override
    public void readFields(DataInput in) throws IOException {
        user = in.readLong();
        mutualFriend = in.readLong();
    }

    @Override
    public String toString() {
        return "toUser: " + user + " mutualFriend: " + mutualFriend;
    }
}
```

### b.**Map （用于生成每个用户的好友推荐记录）**

```java
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.StringTokenizer;

public class Map extends Mapper<LongWritable, Text, LongWritable, FriendCountWritable> implements Tool {
    private Configuration conf;

    @Override
    public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
        String line[] = value.toString().split("\t");

        Long fromUser = Long.parseLong(line[0]);
        List<Long> toUsers = new ArrayList<>();

        if (line.length == 2) {
            StringTokenizer tokenizer = new StringTokenizer(line[1], ",");
            while (tokenizer.hasMoreTokens()) {
                Long toUser = Long.parseLong(tokenizer.nextToken());
                toUsers.add(toUser);
                context.write(new LongWritable(fromUser), new FriendCountWritable(toUser, -1L));
            }

            // Generate combinations of friends and their mutual friends
            for (int i = 0; i < toUsers.size(); i++) {
                for (int j = i + 1; j < toUsers.size(); j++) {
                    context.write(new LongWritable(toUsers.get(i)), new FriendCountWritable(toUsers.get(j), fromUser));
                    context.write(new LongWritable(toUsers.get(j)), new FriendCountWritable(toUsers.get(i), fromUser));
                }
            }
        }
    }

    @Override
    public void setConf(Configuration conf) {
        this.conf = conf;
    }

    @Override
    public Configuration getConf() {
        return conf;
    }

    @Override
    public int run(String[] args) throws Exception {
        if (args.length != 2) {
            System.err.println("Usage: Map <input path> <output path>");
            return -1;
        }

        Job job = Job.getInstance(getConf(), "People You Might Know");
        job.setJarByClass(Map.class);
        job.setMapperClass(Map.class);
        job.setReducerClass(Reduce.class);

        job.setMapOutputKeyClass(LongWritable.class);
        job.setMapOutputValueClass(FriendCountWritable.class);

        job.setOutputKeyClass(LongWritable.class);
        job.setOutputValueClass(Text.class);

        org.apache.hadoop.fs.Path inputPath = new org.apache.hadoop.fs.Path(args[0]);
        org.apache.hadoop.fs.Path outputPath = new org.apache.hadoop.fs.Path(args[1]);

        org.apache.hadoop.mapreduce.lib.input.FileInputFormat.addInputPath(job, inputPath);
        org.apache.hadoop.mapreduce.lib.output.FileOutputFormat.setOutputPath(job, outputPath);

        return job.waitForCompletion(true) ? 0 : 1;
    }

    public static void main(String[] args) throws Exception {
        int exitCode = ToolRunner.run(new Configuration(), new Map(), args);
        System.exit(exitCode);
    }
}
```

### **c. Reduce （用于聚合并输出推荐结果）**

```java
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Reducer;

import java.io.IOException;
import java.util.Map;
import java.util.HashMap;
import java.util.List;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.TreeMap;

public class Reduce extends Reducer<LongWritable, FriendCountWritable, LongWritable, Text> {
    @Override
    public void reduce(LongWritable key, Iterable<FriendCountWritable> values, Context context)
            throws IOException, InterruptedException {
        Map<Long, List<Long>> mutualFriends = new HashMap<>();
        
        // Process each value
        for (FriendCountWritable val : values) {
            Long toUser = val.user;  // 直接访问 public 字段
            Long mutualFriend = val.mutualFriend; // 直接访问 public 字段
            if (mutualFriends.containsKey(toUser)) {
                if (mutualFriend != -1) {
                    mutualFriends.get(toUser).add(mutualFriend);
                }
            } else {
                if (mutualFriend != -1) {
                    mutualFriends.put(toUser, new ArrayList<>(Arrays.asList(mutualFriend)));
                }
            }
        }

        // Sort by number of mutual friends (descending order)
        TreeMap<Long, List<Long>> sortedMutualFriends = new TreeMap<>((key1, key2) -> {
            int size1 = mutualFriends.containsKey(key1) ? mutualFriends.get(key1).size() : 0;
            int size2 = mutualFriends.containsKey(key2) ? mutualFriends.get(key2).size() : 0;
            return Integer.compare(size2, size1); // descending order
        });

        sortedMutualFriends.putAll(mutualFriends);

        // Prepare output
        StringBuilder output = new StringBuilder();
        for (Map.Entry<Long, List<Long>> entry : sortedMutualFriends.entrySet()) {
            output.append(entry.getKey()).append(" (").append(entry.getValue().size())
                    .append(": ").append(entry.getValue()).append("), ");
        }

        // Remove trailing comma and space
        if (output.length() > 2) {
            output.setLength(output.length() - 2);
        }

        context.write(key, new Text(output.toString()));
    }
}
```

编写后与数据放在桌面文件夹中路径类似：

```
PeopleYouMightKnow/
    ├── bin            # 编译输出目录
    ├── src            # 源代码目录
    │   ├── FriendCountWritable.java
    │   ├── Map.java
    │   └── Reduce.java
    └── input.txt      # 输入数据文件
```

### 编译

<img src="/Users/pengzexue/Library/Application Support/typora-user-images/image-20241216012445887.png" style="zoom:50%;" />

### 打包

<img src="/Users/pengzexue/Library/Application Support/typora-user-images/image-20241216012929058.png" style="zoom:50%;" />

验证完整性：

<img src="/Users/pengzexue/Library/Application Support/typora-user-images/image-20241216013130280.png" style="zoom:50%;" />

## 2.启动 Hadoop 集群（分布式文件系统）

以下步骤适用伪分布式环境：

**启动 Hadoop 服务**,执行以下命令启动 Hadoop 集群，包括 HDFS 和 YARN：

<img src="/Users/pengzexue/Library/Application Support/typora-user-images/image-20241216000155900.png" style="zoom:50%;" />

之后启动yarn出现错误首先配置yarn文件将日志文件储存在日志目录 /opt/homebrew/opt/hadoop/log中在查看运行后的日志文件

<img src="/Users/pengzexue/Library/Application Support/typora-user-images/image-20241216004244022.png" style="zoom:50%;" />

​	从日志信息来看，ResourceManager 启动失败的主要原因是 **Java 模块限制**，导致 java.lang.reflect.InaccessibleObjectException 异常。这种问题常见于 Hadoop 与较新的 Java 版本（如 Java 17 或更高版本）之间的不兼容。

所以要么降级java到底版本或者继续修改yarn配置文件：

```cmd
$ nano /opt/homebrew/opt/hadoop/libexec/etc/hadoop/yarn-env.sh
```

更改配置到如下，设置java变量：

<img src="/Users/pengzexue/Library/Application Support/typora-user-images/image-20241216004538251.png" style="zoom:50%;" />

解决问题之后

## 3.启动yarn（资源调度）

```cmd
start-yarn.sh
```

查看运行情况：

<img src="/Users/pengzexue/Library/Application Support/typora-user-images/image-20241216004840101.png" style="zoom:50%;" />

​	**检查 HDFS 状态**：打开浏览器，访问 http://localhost:9870，查看 HDFS Web UI。

<img src="/Users/pengzexue/Library/Application Support/typora-user-images/image-20241216005251319.png" style="zoom:50%;" />

​	**检查 YARN 状态**：打开浏览器，访问 http://localhost:8088，查看 YARN 资源管理界面。

<img src="/Users/pengzexue/Library/Application Support/typora-user-images/image-20241216005350689.png" style="zoom:50%;" />

## 4.**上传数据到 HDFS**

### **a.创建 HDFS 目录**

<img src="/Users/pengzexue/Library/Application Support/typora-user-images/image-20241216005727175.png" style="zoom:50%;" />

### b.**上传数据文件到 HDFS**

```cmd
hdfs dfs -put /Users/pengzexue/Desktop/PeopleYouMightKnow/input.txt /user/input/
```

### c. **验证数据是否成功上传**

<img src="/Users/pengzexue/Library/Application Support/typora-user-images/image-20241216010354053.png" style="zoom:50%;" />

 **运行 MapReduce 作业**

```cmd
hadoop jar /Users/pengzexue/Desktop/PeopleYouMightKnow/FriendRecommendation.jar Map /user/input /user/output 
```

<img src="/Users/pengzexue/Library/Application Support/typora-user-images/image-20241216014142642.png" style="zoom:50%;" />

查看输出（图片展示部分输出）

```cmd
hdfs dfs -cat /user/output/part-r-00000
```

<img src="/Users/pengzexue/Library/Application Support/typora-user-images/image-20241216014405306.png" style="zoom:50%;" />

输出结果下载到本地：

```cmd
hdfs dfs -get /user/output/part-r-00000 /Users/pengzexue/Desktop/PeopleYouMightKnow
```

