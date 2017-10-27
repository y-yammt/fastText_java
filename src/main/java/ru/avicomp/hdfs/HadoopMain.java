package ru.avicomp.hdfs;

import java.io.IOException;
import java.io.InputStream;
import java.net.URI;
import java.net.URISyntaxException;
import java.net.URL;
import java.nio.file.FileSystemNotFoundException;
import java.nio.file.Paths;
import java.time.Duration;
import java.time.Instant;
import java.time.temporal.ChronoUnit;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.function.Function;
import java.util.stream.Collectors;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;

import cc.fasttext.Args;
import cc.fasttext.FastText;
import cc.fasttext.Main;
import ru.avicomp.io.IOStreams;
import ru.avicomp.io.InputStreamSupplier;
import ru.avicomp.io.OutputStreamSupplier;

/**
 * TODO: for now it is just test (train model).
 * Example of command:
 * <pre>{@code
 * -hadoop-url hdfs://172.16.35.1:54310 -hadoop-user hadoop cbow -thread 4 -dim 128 -ws 5 -epoch 10 -minCount 5
 * -input /tmp/out/raw-text-data.txt -output /tmp/out/raw-text-data.win.cbox.d128.w5.hs
 * }</pre>
 * Created by @szuev on 25.10.2017.
 */
public class HadoopMain {
    private static final String HADOOP_URL_KEY = "-hadoop-url";
    private static final String HADOOP_USER_KEY = "-hadoop-user";
    private static final String FS_INPUT = "-input";
    private static final String FS_OUTPUT = "-output";
    private static final String HADOOP_SETTINGS_PREFIX = "-hadoop-";
    private static final String PROPERTIES_PREFIX = "-prop-";

    public static void main(String... array) throws Exception {
        Instant t0 = Instant.now();
        System.out.println("Start: " + t0);
        List<String> list = Arrays.stream(array).collect(Collectors.toList());
        String hadoopURL = takeArg(list, HADOOP_URL_KEY);
        String hadoopUser = takeArg(list, HADOOP_USER_KEY);
        Map<String, String> hadoopSettings = list.stream().filter(s -> s.startsWith(HADOOP_SETTINGS_PREFIX))
                .collect(Collectors.toSet()).stream().collect(Collectors.toMap(Function.identity(), k -> takeArg(list, k)));

        Map<String, String> properties = list.stream().filter(s -> s.startsWith(PROPERTIES_PREFIX))
                .collect(Collectors.toSet()).stream().collect(Collectors.toMap(Function.identity(), k -> takeArg(list, k)));

        String input = getArg(list, FS_INPUT);
        String output = getArg(list, FS_OUTPUT);

        System.out.println(String.format("hadoop-url: <%s>, hadoop-user: %s, input=<%s>, output=<%s>, hadoop-map: %s", hadoopURL, hadoopUser, input, output, hadoopSettings));
        System.out.println("Rest args: " + list);
        IOStreams fs = createHadoopFS(hadoopURL, hadoopUser, hadoopSettings, properties);

        Args args = Main.parseArgs(list.toArray(new String[list.size()])).setIOStreams(fs);
        FastText fasttext = new FastText(args);
        fasttext.train();
        Instant t1 = Instant.now();
        Duration duration = Duration.between(t0, t1);
        float seconds = duration.get(ChronoUnit.SECONDS) + duration.get(ChronoUnit.NANOS) / 1_000_000_000f;
        System.out.println(String.format("Time: %s%s.", seconds > 60 ? seconds / 60 : seconds, seconds > 60 ? "m" : "s"));
    }

    public static IOStreams createHadoopFS(String url, String user, Map<String, String> settings) throws IOException {
        return createHadoopFS(url, user, settings, Collections.emptyMap());
    }

    public static IOStreams createHadoopFS(String url, String user, Map<String, String> settings, Map<String, String> props) throws IOException {
        Configuration conf = new Configuration();
        conf.set("fs.defaultFS", url);
        conf.set("fs.hdfs.impl", org.apache.hadoop.hdfs.DistributedFileSystem.class.getName());
        conf.set("fs.file.impl", org.apache.hadoop.fs.LocalFileSystem.class.getName());
        settings.forEach(conf::set);
        System.setProperty("HADOOP_USER_NAME", user);
        String home;
        try {
            URL resource = HadoopMain.class.getResource("/bin");
            System.out.println(resource);
            home = Paths.get(resource.toURI()).getParent().toString();
        } catch (URISyntaxException | FileSystemNotFoundException e) { // if jar
            home = "/";
        }
        System.setProperty("hadoop.home.dir", home);
        props.forEach(System::setProperty);

        FileSystem fs = FileSystem.get(URI.create(url), conf);
        return new IOStreams() {
            @Override
            public String toString() {
                return String.format("Hadoop-FS: %s@<%s>%s%s", user, url, settings, props);
            }

            @Override
            public InputStreamSupplier createInput(String path) {
                Path in = new Path(path);
                return new InputStreamSupplier() {
                    @Override
                    public InputStream open() throws IOException {
                        return fs.open(in);
                    }

                    @Override
                    public long bytes() throws IOException, UnsupportedOperationException {
                        return fs.getFileStatus(in).getLen();
                    }
                };
            }

            @Override
            public OutputStreamSupplier createOutput(String path) {
                return () -> fs.create(new Path(path));
            }
        };
    }

    private static String takeArg(List<String> args, String key) {
        if (!args.contains(key)) throw new IllegalArgumentException("Mandatory arg: " + key);
        int index = args.indexOf(key);
        String res = args.get(index + 1);
        args.remove(index);
        args.remove(index);
        return res;
    }

    private static String getArg(List<String> args, String key) {
        if (!args.contains(key)) throw new IllegalArgumentException("Mandatory arg: " + key);
        int index = args.indexOf(key);
        return args.get(index + 1);
    }

}
