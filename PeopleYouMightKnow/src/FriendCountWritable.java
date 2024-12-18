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