package cc.fasttext;

import java.nio.charset.StandardCharsets;
import java.util.Map;

import org.junit.Test;

import static org.junit.Assert.assertEquals;

public class DictionaryTest {

    private Dictionary dictionary = new Dictionary(new Args.Builder().build(), StandardCharsets.UTF_8);

	@Test
	public void testHash() {
		assertEquals(dictionary.hash(","), 688690635L);
		assertEquals(dictionary.hash("is"), 1312329493L);
		assertEquals(dictionary.hash("</s>"), 3617362777L);
	}

	@Test
	public void testFind() {
		assertEquals(dictionary.find(","), 28690635L);
		assertEquals(dictionary.find("is"), 22329493L);
		assertEquals(dictionary.find("</s>"), 17362777L);
	}

	@Test
	public void testAdd() {
		dictionary.add(",");
		dictionary.add("is");
		dictionary.add("is");
		String w = "";
		dictionary.add(w);
		dictionary.add(w);
		dictionary.add(w);
		Map<Long, Integer> word2int = dictionary.getWord2int();
		assertEquals(3, dictionary.getWords().get(word2int.get(dictionary.find(w))).count());
		assertEquals(2, dictionary.getWords().get(word2int.get(dictionary.find("is"))).count());
		assertEquals(1, dictionary.getWords().get(word2int.get(dictionary.find(","))).count());
	}
}
