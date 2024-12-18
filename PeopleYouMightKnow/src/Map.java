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