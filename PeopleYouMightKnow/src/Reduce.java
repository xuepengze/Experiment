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