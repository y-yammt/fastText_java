package cc.fasttext;

import org.apache.commons.math3.random.Well19937c;
import org.junit.Assert;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Arrays;
import java.util.List;
import java.util.stream.Stream;

/**
 * Created by @szuev on 05.12.2017.
 */
public class MatrixTest {
    private static final Logger LOGGER = LoggerFactory.getLogger(MatrixTest.class);

    @Test
    public void testQuantize() {
        int mSize = 300;
        int nSize = 10;
        Matrix m = new Matrix(mSize, nSize);
        for (int i = 0; i < mSize; i++) {
            for (int j = 0; j < nSize; j++) {
                m.set(i, j, i * mSize + j);
            }
        }

        LOGGER.info("{}", m);
        Assert.assertEquals(301, m.get(1, 1), 0);
        Assert.assertEquals(3309, m.get(11, 9), 0);
        Assert.assertEquals(2708, m.get(9, 8), 0);

        QMatrix q = new QMatrix(m
                , Well19937c::new
                , 2, true);
        LOGGER.info("{}", q);
        Stream.of(q.getPQ(), q.getNPQ()).forEach(pq -> {
            LOGGER.debug("{}::{}", pq.getCentroids().size(), pq.getCentroids());
        });
        Assert.assertEquals(2560, q.getPQ().getCentroids().size());
        Assert.assertEquals(256, q.getNPQ().getCentroids().size());

        Vector v = new Vector(nSize);
        LOGGER.info("{}", v);
        v.addRow(q, 12);
        LOGGER.info("{}", v);
        List<Float> expected = Arrays.asList(3600F, 3601F, 3601.9F, 3602.9F, 3604F, 3605F, 3606F, 3607F, 3608F, 3609F);
        List<Float> actual = v.getData();
        Assert.assertEquals(expected.size(), actual.size());
        for (int i = 0; i < actual.size(); i++) {
            Assert.assertEquals("Wrong #" + i, expected.get(i), actual.get(i), 0.2);
        }
    }
}
