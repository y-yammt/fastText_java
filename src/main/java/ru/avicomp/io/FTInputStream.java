package ru.avicomp.io;

import java.io.FilterInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.Objects;

import com.google.common.io.LittleEndianDataInputStream;

/**
 * FastText InputStream.
 *
 * To read byte data in cpp little endian style.
 * Covers only primitives.
 *
 * @see com.google.common.io.LittleEndianDataInputStream
 * Created by @szuev on 26.10.2017.
 */
public class FTInputStream extends FilterInputStream {

    public FTInputStream(InputStream in) {
        super(wrap(in));
    }

    private static LittleEndianDataInputStream wrap(InputStream in) {
        Objects.requireNonNull(in, "Null input stream specified");
        return in instanceof LittleEndianDataInputStream ? (LittleEndianDataInputStream) in : new LittleEndianDataInputStream(in);
    }

    private LittleEndianDataInputStream in() {
        return (LittleEndianDataInputStream) in;
    }

    public void readFully(byte[] b) throws IOException {
        in().readFully(b);
    }

    public void readFully(byte[] b, int off, int len) throws IOException {
        in().readFully(b, off, len);
    }

    public boolean readBoolean() throws IOException {
        return in().readBoolean();
    }

    public byte readByte() throws IOException {
        return in().readByte();
    }

    public int readUnsignedByte() throws IOException {
        return in().readUnsignedByte();
    }

    public short readShort() throws IOException {
        return in().readShort();
    }

    public int readUnsignedShort() throws IOException {
        return in().readUnsignedShort();
    }

    public char readChar() throws IOException {
        return in().readChar();
    }

    public int readInt() throws IOException {
        return in().readInt();
    }

    public long readLong() throws IOException {
        return in().readLong();
    }

    public float readFloat() throws IOException {
        return in().readFloat();
    }

    public double readDouble() throws IOException {
        return in().readDouble();
    }

}
